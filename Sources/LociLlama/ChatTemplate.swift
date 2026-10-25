import Foundation
import llama

public enum ChatRole: String, Sendable, Equatable {
    case system, user, assistant
}

public struct ChatMessage: Sendable, Equatable {
    public var role: ChatRole
    public var content: String

    public init(role: ChatRole, content: String) {
        self.role = role
        self.content = content
    }
}

/// Renders a conversation into the text a chat model expects.
///
/// llama.cpp's built-in applier recognises a fixed list of template layouts (not Jinja), so it is
/// tried first. Layouts it does not know are rendered by hand when they belong to a supported
/// family; anything else is reported instead of producing a prompt the model was not trained on.
enum ChatTemplate {
    struct Unsupported: Error, Equatable {}

    /// The prompt for `messages`, ending where the assistant's reply begins.
    ///
    /// - Parameters:
    ///   - template: the model's embedded Jinja template text.
    ///   - bos: the model's beginning-of-sequence text, used by layouts that start with it.
    static func render(_ messages: [ChatMessage], template: String, bos: String) throws -> String {
        var prompt: String
        if let builtin = applyBuiltin(messages, template: template) {
            prompt = builtin
        } else if GemmaTurns.matches(template) {
            prompt = GemmaTurns.render(messages, bos: bos)
        } else {
            throw Unsupported()
        }
        // Qwen3-style templates think by default and switch it off with an empty block.
        if template.contains("enable_thinking"), template.contains("<think>") {
            prompt += "<think>\n\n</think>\n\n"
        }
        return prompt
    }

    private static func applyBuiltin(_ messages: [ChatMessage], template: String) -> String? {
        var roles: [UnsafeMutablePointer<CChar>] = []
        var contents: [UnsafeMutablePointer<CChar>] = []
        defer {
            roles.forEach { free($0) }
            contents.forEach { free($0) }
        }
        for message in messages {
            guard let role = strdup(message.role.rawValue), let content = strdup(message.content) else { return nil }
            roles.append(role)
            contents.append(content)
        }
        var chat = zip(roles, contents).map { llama_chat_message(role: UnsafePointer($0), content: UnsafePointer($1)) }

        var capacity = messages.reduce(1024) { $0 + $1.content.utf8.count * 2 + 64 }
        for _ in 0..<2 {
            var buffer = [CChar](repeating: 0, count: capacity)
            let length = llama_chat_apply_template(template, &chat, chat.count, true, &buffer, Int32(capacity))
            if length < 0 { return nil }
            if Int(length) <= capacity {
                return String(decoding: buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
            }
            capacity = Int(length) + 1
        }
        return nil
    }
}

/// The `<|turn>role … <turn|>` layout of Gemma 4, with `model` as the assistant role. Verified
/// against the model's own Jinja template (see `ChatTemplateTests`).
enum GemmaTurns {
    static func matches(_ template: String) -> Bool {
        template.contains("<|turn>") && template.contains("<turn|>")
    }

    static func render(_ messages: [ChatMessage], bos: String) -> String {
        var text = bos
        var rest = messages[...]
        if let first = rest.first, first.role == .system {
            text += "<|turn>system\n" + trimmed(first.content) + "<turn|>\n"
            rest = rest.dropFirst()
        }
        for message in rest {
            switch message.role {
            case .system, .user:
                text += "<|turn>\(message.role.rawValue)\n" + trimmed(message.content) + "<turn|>\n"
            case .assistant:
                text += "<|turn>model\n" + strippingThinking(message.content) + "<turn|>\n"
            }
        }
        return text + "<|turn>model\n"
    }

    private static func trimmed(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Drops `<|channel>…<channel|>` reasoning from an earlier reply, as the template does.
    static func strippingThinking(_ text: String) -> String {
        var result = ""
        for part in text.components(separatedBy: "<channel|>") {
            if let open = part.range(of: "<|channel>") {
                result += part[..<open.lowerBound]
            } else {
                result += part
            }
        }
        return trimmed(result)
    }
}
