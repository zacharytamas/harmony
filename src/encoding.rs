use crate::{
    chat::{Author, Content, Message, ReasoningEffort, Role, SystemContent, TextContent},
    tiktoken::{CoreBPE, Rank},
};
use anyhow::Context as _;
use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    vec,
};

// Parsed representation of a message header.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParsedHeader {
    author: Author,
    recipient: Option<String>,
    channel: Option<String>,
    content_type: Option<String>,
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum RenderFormattingTokenError {
    #[error("tried to render unmapped formatting token {0}")]
    UnmappedToken(FormattingToken),

    #[error(
        "Expected encoding of formatting token {token} to be a single token, but got {encoding:?}"
    )]
    InvalidEncoding {
        token: FormattingToken,
        encoding: Vec<Rank>,
    },
}

/// These are formatting tokens that the renderer can use to generically
/// format the output of the model, but at formatting time, they are replaced
/// by actual tokens from the tokenizers vocabulary.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FormattingToken {
    Start,
    Message,
    EndMessage,
    EndMessageDoneSampling,
    EndMessageAssistantToTool,
    Refusal,
    ConstrainedFormat,
    Channel,
    BeginUntrusted,
    EndUntrusted,
    MetaSep,
    MetaEnd,
}

impl FormattingToken {
    fn as_str(&self) -> &str {
        match self {
            FormattingToken::Start => "<|start|>",
            FormattingToken::Message => "<|message|>",
            FormattingToken::EndMessage => "<|end|>",
            FormattingToken::EndMessageDoneSampling => "<|return|>",
            FormattingToken::EndMessageAssistantToTool => "<|call|>",
            FormattingToken::Refusal => "<|refusal|>",
            FormattingToken::ConstrainedFormat => "<|constrain|>",
            FormattingToken::Channel => "<|channel|>",
            FormattingToken::BeginUntrusted => "<|untrusted|>",
            FormattingToken::EndUntrusted => "<|end_untrusted|>",
            FormattingToken::MetaSep => "<|channel|>",
            FormattingToken::MetaEnd => "<|meta_end|>",
        }
    }
}

impl std::fmt::Display for FormattingToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct HarmonyEncoding {
    pub(crate) name: String,
    pub(crate) n_ctx: usize,
    pub(crate) max_message_tokens: usize,
    pub(crate) max_action_length: usize,
    pub(crate) tokenizer_name: String,
    pub(crate) tokenizer: Arc<CoreBPE>,
    pub(crate) format_token_mapping: HashMap<FormattingToken, String>,
    pub(crate) stop_formatting_tokens: HashSet<FormattingToken>,
    pub(crate) stop_formatting_tokens_for_assistant_actions: HashSet<FormattingToken>,
    pub(crate) conversation_has_function_tools: Arc<AtomicBool>,
}

impl std::fmt::Debug for HarmonyEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HarmonyEncoding")
            .field("name", &self.name)
            .field("tokenizer_name", &self.tokenizer_name)
            .field("n_ctx", &self.n_ctx)
            .field("max_message_tokens", &self.max_message_tokens)
            .field("max_action_length", &self.max_action_length)
            .finish()
    }
}

impl std::fmt::Display for HarmonyEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Renderer({})", self.name)
    }
}

// General methods
impl HarmonyEncoding {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tokenizer_name(&self) -> &str {
        &self.tokenizer_name
    }

    pub fn max_message_tokens(&self) -> usize {
        self.max_message_tokens
    }

    pub fn tokenizer(&self) -> &CoreBPE {
        &self.tokenizer
    }

    pub fn stop_tokens(&self) -> anyhow::Result<HashSet<Rank>> {
        self.stop_formatting_tokens
            .iter()
            .copied()
            .map(|t| match self.render_formatting_token(t) {
                Ok(t) => Ok(t),
                Err(RenderFormattingTokenError::UnmappedToken(_)) => Err(anyhow::anyhow!(
                    "token {t} was specified as a stop token, but is not mapped"
                )),
                Err(e) => Err(anyhow::anyhow!(e).context("could not render stop token")),
            })
            .collect()
    }

    pub fn stop_tokens_for_assistant_actions(&self) -> anyhow::Result<HashSet<Rank>> {
        self.stop_formatting_tokens_for_assistant_actions
            .iter()
            .copied()
            .map(|t| match self.render_formatting_token(t) {
                Ok(t) => Ok(t),
                Err(RenderFormattingTokenError::UnmappedToken(_)) => Err(anyhow::anyhow!(
                    "token {t} was specified as a stop token, but is not mapped"
                )),
                Err(e) => Err(anyhow::anyhow!(e).context("could not render stop token")),
            })
            .collect()
    }
}

// Methods for rendering conversations
impl HarmonyEncoding {
    /// Renders a conversation into a collection of tokens.
    pub fn render_conversation_into<'a, I, B>(
        &self,
        conversation: I,
        into: &mut B,
        config: Option<&RenderConversationConfig>,
    ) -> anyhow::Result<()>
    where
        I: IntoIterator<Item = &'a Message>,
        B: Extend<Rank>,
    {
        let messages: Vec<_> = conversation.into_iter().collect();
        let has_function_tools = messages.iter().any(|msg| {
            msg.content.iter().any(|c| {
                if let Content::DeveloperContent(dev) = c {
                    if let Some(tools) = &dev.tools {
                        if let Some(ns) = tools.get("functions") {
                            !ns.tools.is_empty()
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            })
        });
        self.conversation_has_function_tools
            .store(has_function_tools, Ordering::Relaxed);
        let last_assistant_is_final = messages
            .iter()
            .rev()
            .find_map(|msg| {
                (msg.author.role == Role::Assistant)
                    .then(|| msg.channel.as_deref() == Some("final"))
            })
            .unwrap_or(false);

        let should_drop_analysis =
            config.is_some_and(|c| c.auto_drop_analysis && last_assistant_is_final);

        let first_final_idx = messages
            .iter()
            .position(|msg| msg.channel.as_deref() == Some("final"));

        let result = messages
            .iter()
            .enumerate()
            .filter(|(idx, msg)| {
                !(should_drop_analysis
                    && first_final_idx.is_some_and(|first| *idx < first)
                    && msg.channel.as_deref() == Some("analysis"))
            })
            .try_for_each(|(_, msg)| self.render_into(*msg, into));
        self.conversation_has_function_tools
            .store(false, Ordering::Relaxed);
        result?;
        Ok(())
    }

    /// Renders a conversation into a collection of tokens, adding the next turn role.
    ///
    /// This method is used to prepare a conversation for inference.
    pub fn render_conversation_for_completion_into<'a, I, B>(
        &self,
        conversation: I,
        next_turn_role: Role,
        into: &mut B,
        config: Option<&RenderConversationConfig>,
    ) -> anyhow::Result<()>
    where
        I: IntoIterator<Item = &'a Message>,
        B: Extend<Rank>,
    {
        let _config = config.unwrap_or(&RenderConversationConfig::default());
        self.render_conversation_into(conversation, into, config)?;
        self.render_formatting_token_into(FormattingToken::Start, into)?;
        self.render_text_into(next_turn_role.as_str(), into)?;
        Ok(())
    }

    pub fn render_conversation_for_completion<'a, I>(
        &self,
        conversation: I,
        next_turn_role: Role,
        config: Option<&RenderConversationConfig>,
    ) -> anyhow::Result<Vec<Rank>>
    where
        I: IntoIterator<Item = &'a Message>,
    {
        let mut into = vec![];
        self.render_conversation_for_completion_into(
            conversation,
            next_turn_role,
            &mut into,
            config,
        )?;
        Ok(into)
    }

    /// Render a conversation for training.
    ///
    /// If the last message in the conversation is an assistant message to the
    /// `final` channel, replace the trailing `<|end|>` token with `<|return|>`.
    pub fn render_conversation_for_training<'a, I>(
        &self,
        conversation: I,
        config: Option<&RenderConversationConfig>,
    ) -> anyhow::Result<Vec<Rank>>
    where
        I: IntoIterator<Item = &'a Message>,
    {
        let messages: Vec<&Message> = conversation.into_iter().collect();
        let mut out = vec![];
        self.render_conversation_into(messages.iter().copied(), &mut out, config)?;
        if let Some(last) = messages.last() {
            if last.author.role == Role::Assistant && last.channel.as_deref() == Some("final") {
                if let Some(last_token) = out.last_mut() {
                    *last_token =
                        self.render_formatting_token(FormattingToken::EndMessageDoneSampling)?;
                }
            }
        }
        Ok(out)
    }

    /// Render a conversation without appending a new role.
    pub fn render_conversation<'a, I>(
        &self,
        conversation: I,
        config: Option<&RenderConversationConfig>,
    ) -> anyhow::Result<Vec<Rank>>
    where
        I: IntoIterator<Item = &'a Message>,
    {
        let mut out = vec![];
        self.render_conversation_into(conversation, &mut out, config)?;
        Ok(out)
    }

    /// Render a single message into tokens.
    pub fn render(&self, message: &Message) -> anyhow::Result<Vec<Rank>> {
        let mut out = vec![];
        Render::<Message>::render(self, message, &mut out)?;
        Ok(out)
    }

    /// Render a single message into the provided buffer.
    pub fn render_into<B>(&self, message: &Message, into: &mut B) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        Render::<Message>::render(self, message, into)
    }
}

// Rendering helper methods
impl HarmonyEncoding {
    fn mapped_format_token(&self, t: FormattingToken) -> Option<&str> {
        self.format_token_mapping.get(&t).map(|s| s.as_str())
    }

    fn render_formatting_token(
        &self,
        t: FormattingToken,
    ) -> Result<Rank, RenderFormattingTokenError> {
        let mapped = self
            .mapped_format_token(t)
            .ok_or(RenderFormattingTokenError::UnmappedToken(t))?;
        let encoded = self.tokenizer.encode_with_special_tokens(mapped);
        if encoded.len() != 1 {
            return Err(RenderFormattingTokenError::InvalidEncoding {
                token: t,
                encoding: encoded,
            });
        }
        Ok(encoded[0])
    }

    fn render_formatting_token_into<B>(
        &self,
        t: FormattingToken,
        into: &mut B,
    ) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        let r = self.render_formatting_token(t)?;
        into.extend(std::iter::once(r));
        Ok(())
    }

    fn render_text_into<T, B>(&self, text: T, into: &mut B) -> anyhow::Result<()>
    where
        T: AsRef<str>,
        B: Extend<Rank>,
    {
        into.extend(self.tokenizer.encode_ordinary(text.as_ref()));
        Ok(())
    }

    pub fn parse_messages_from_completion_tokens<I>(
        &self,
        tokens: I,
        mut role: Option<Role>,
    ) -> anyhow::Result<Vec<Message>>
    where
        I: IntoIterator<Item = Rank>,
    {
        let mut messages = Vec::<Message>::new();
        let mut parser = Parser {
            encoding: self,
            tokens: tokens.into_iter().peekable(),
        };
        loop {
            let (message, did_reach_end_of_stream) = parser.parse_message(role)?;
            messages.push(message);
            role = None;
            if did_reach_end_of_stream {
                break;
            }
        }
        anyhow::ensure!(parser.tokens.next().is_none(), "Expected end of stream");
        Ok(messages)
    }

    /// Helper to convert a JSON schema (OpenAPI style) to a TypeScript type definition.
    fn json_schema_to_typescript(schema: &serde_json::Value, indent: &str) -> String {
        // Helper to check if this schema is an enum
        fn is_enum(schema: &serde_json::Value) -> bool {
            return schema
                .get("enum")
                .and_then(|e| e.as_array())
                .map_or(false, |arr| !arr.is_empty());
        }

        // Handle oneOf at the top level
        if let Some(one_of) = schema.get("oneOf") {
            if let Some(arr) = one_of.as_array() {
                let mut out = String::new();
                let mut first = true;
                for variant in arr {
                    if !first {
                        out.push_str("\n");
                        out.push_str(&format!("{} | ", indent));
                    } else {
                        out.push_str(&format!("\n{} | ", indent));
                        first = false;
                    }
                    let type_str =
                        Self::json_schema_to_typescript(variant, &format!("{}   ", indent));
                    let mut type_str = type_str;
                    if variant
                        .get("nullable")
                        .and_then(|n| n.as_bool())
                        .unwrap_or(false)
                    {
                        if !type_str.contains("null") {
                            type_str = format!("{} | null", type_str);
                        }
                    }
                    out.push_str(&type_str);
                    // Add trailing comments (description, default)
                    let mut trailing_comments = Vec::new();
                    if let Some(desc) = variant.get("description") {
                        if let Some(desc_str) = desc.as_str() {
                            trailing_comments.push(format!("{}", desc_str));
                        }
                    }
                    if let Some(default) = variant.get("default") {
                        if default.is_string() && !is_enum(variant) {
                            trailing_comments
                                .push(format!("default: \"{}\"", default.as_str().unwrap()));
                        } else {
                            trailing_comments.push(format!("default: {}", default));
                        }
                    }
                    if !trailing_comments.is_empty() {
                        out.push_str(&format!(" // {}", trailing_comments.join(" ")));
                    }
                }
                return out;
            }
        }
        // Handle type as array (e.g., ["number", "string"])
        if let Some(types) = schema.get("type").and_then(|v| v.as_array()) {
            let mut type_strings = Vec::new();
            for ty in types {
                if let Some(ty_str) = ty.as_str() {
                    let mapped = match ty_str {
                        "integer" => "number",
                        other => other,
                    };
                    type_strings.push(mapped.to_string());
                }
            }
            if !type_strings.is_empty() {
                return type_strings.join(" | ");
            }
        }
        // Handle type
        if let Some(ty) = schema.get("type").and_then(|v| v.as_str()) {
            match ty {
                "object" => {
                    let mut out = String::new();
                    // Render object-level description as comment
                    if let Some(desc) = schema.get("description") {
                        if let Some(desc_str) = desc.as_str() {
                            out.push_str(&format!("{}// {}\n", indent, desc_str));
                        }
                    }
                    out.push_str("{\n");

                    if let Some(props) = schema.get("properties") {
                        if let Some(props_map) = props.as_object() {
                            // Determine required fields
                            let mut required = std::collections::HashSet::new();
                            if let Some(req) = schema.get("required") {
                                if let Some(req_arr) = req.as_array() {
                                    for r in req_arr {
                                        if let Some(s) = r.as_str() {
                                            required.insert(s);
                                        }
                                    }
                                }
                            }
                            for (key, val) in props_map {
                                // Render title, description, and examples as comments
                                if let Some(title) = val.get("title") {
                                    if let Some(title_str) = title.as_str() {
                                        out.push_str(&format!(
                                            "{0}// {1}\n{0}//\n",
                                            indent, title_str
                                        ));
                                    }
                                }
                                // Only render description here if not a oneOf property
                                if val.get("oneOf").is_none() {
                                    if let Some(desc) = val.get("description") {
                                        if let Some(desc_str) = desc.as_str() {
                                            out.push_str(&format!("{}// {}\n", indent, desc_str));
                                        }
                                    }
                                }
                                if let Some(examples) = val.get("examples") {
                                    if let Some(arr) = examples.as_array() {
                                        if !arr.is_empty() {
                                            out.push_str(&format!("{}// Examples:\n", indent));
                                            for ex in arr {
                                                if let Some(ex_str) = ex.as_str() {
                                                    out.push_str(&format!(
                                                        "{}// - \"{}\"\n",
                                                        indent, ex_str
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                                // Handle oneOf at the property level
                                if let Some(one_of) = val.get("oneOf") {
                                    if let Some(arr) = one_of.as_array() {
                                        // Deduplicate property-level description if it matches the first variant's description
                                        let mut property_desc: Option<&str> = None;
                                        if let Some(desc) = val.get("description") {
                                            if let Some(desc_str) = desc.as_str() {
                                                property_desc = Some(desc_str);
                                            }
                                        }
                                        let mut skip_property_desc = false;
                                        if let Some(desc_str) = property_desc {
                                            if let Some(first_variant) = arr.get(0) {
                                                if let Some(variant_desc) =
                                                    first_variant.get("description")
                                                {
                                                    if let Some(variant_desc_str) =
                                                        variant_desc.as_str()
                                                    {
                                                        if desc_str == variant_desc_str {
                                                            skip_property_desc = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        // Add property-level comments above the property name if not skipped
                                        let mut rendered_property_desc_above = false;
                                        if !skip_property_desc {
                                            if let Some(desc_str) = property_desc {
                                                out.push_str(&format!(
                                                    "{}// {}\n",
                                                    indent, desc_str
                                                ));
                                                rendered_property_desc_above = true;
                                            }
                                        }
                                        if let Some(default) = val.get("default") {
                                            if default.is_string() && !is_enum(val) {
                                                out.push_str(&format!(
                                                    "{}// default: \"{}\"\n",
                                                    indent,
                                                    default.as_str().unwrap()
                                                ));
                                            } else if default.is_string() {
                                                out.push_str(&format!(
                                                    "{}// default: {}\n",
                                                    indent,
                                                    default.as_str().unwrap()
                                                ));
                                            } else {
                                                out.push_str(&format!(
                                                    "{}// default: {}\n",
                                                    indent, default
                                                ));
                                            }
                                        }
                                        // Add property name and optional marker
                                        out.push_str(&format!(
                                            "{}{}{}:\n",
                                            indent,
                                            key,
                                            if required.contains(key.as_str()) {
                                                ""
                                            } else {
                                                "?"
                                            }
                                        ));
                                        // Render each variant
                                        for (i, variant) in arr.iter().enumerate() {
                                            out.push_str(&format!("{} | ", indent));
                                            let type_str = Self::json_schema_to_typescript(
                                                variant,
                                                &format!("{}   ", indent),
                                            );
                                            // Handle nullable in variant
                                            let mut type_str = type_str;
                                            if variant
                                                .get("nullable")
                                                .and_then(|n| n.as_bool())
                                                .unwrap_or(false)
                                            {
                                                if !type_str.contains("null") {
                                                    type_str = format!("{} | null", type_str);
                                                }
                                            }
                                            out.push_str(&type_str);
                                            // Add variant-level comments after the type
                                            let mut trailing_comments = Vec::new();
                                            if i == 0 && rendered_property_desc_above {
                                                // Do not add any description for the first variant if property-level description was rendered above
                                            } else if let Some(desc) = variant.get("description") {
                                                if let Some(desc_str) = desc.as_str() {
                                                    // Only render if not equal to property-level description
                                                    if Some(desc_str) != property_desc {
                                                        trailing_comments
                                                            .push(format!("{}", desc_str));
                                                    }
                                                }
                                            }
                                            if let Some(default) = variant.get("default") {
                                                if default.is_string() && !is_enum(variant) {
                                                    trailing_comments.push(format!(
                                                        "default: \"{}\"",
                                                        default.as_str().unwrap()
                                                    ));
                                                } else if default.is_string() {
                                                    trailing_comments.push(format!(
                                                        "default: {}",
                                                        default.as_str().unwrap()
                                                    ));
                                                } else {
                                                    trailing_comments
                                                        .push(format!("default: {}", default));
                                                }
                                            }
                                            if !trailing_comments.is_empty() {
                                                out.push_str(&format!(
                                                    " // {}",
                                                    trailing_comments.join(" ")
                                                ));
                                            }
                                            out.push_str("\n");
                                        }
                                        out.push_str(&format!("{},\n", indent));
                                        continue;
                                    }
                                }
                                // Normal property rendering
                                out.push_str(&format!(
                                    "{}{}{}: ",
                                    indent,
                                    key,
                                    if required.contains(key.as_str()) {
                                        ""
                                    } else {
                                        "?"
                                    }
                                ));
                                // Handle nullable
                                let mut type_str = Self::json_schema_to_typescript(
                                    val,
                                    &format!("{}    ", indent),
                                );
                                if val
                                    .get("nullable")
                                    .and_then(|n| n.as_bool())
                                    .unwrap_or(false)
                                {
                                    if !type_str.contains("null") {
                                        type_str = format!("{} | null", type_str);
                                    }
                                }
                                out.push_str(&type_str);
                                out.push_str(",");
                                // Add default as comment if present (and not already handled)
                                if val.get("oneOf").is_none() {
                                    if let Some(default) = val.get("default") {
                                        if default.is_string() && !is_enum(val) {
                                            out.push_str(&format!(
                                                " // default: \"{}\"",
                                                default.as_str().unwrap()
                                            ));
                                        } else if default.is_string() {
                                            out.push_str(&format!(
                                                " // default: {}",
                                                default.as_str().unwrap()
                                            ));
                                        } else {
                                            out.push_str(&format!(" // default: {}", default));
                                        }
                                    }
                                }
                                out.push_str("\n");
                            }
                        }
                    }
                    out.push_str(&format!("{}}}", indent));
                    out
                }
                "string" => {
                    if let Some(enum_vals) = schema.get("enum") {
                        if let Some(arr) = enum_vals.as_array() {
                            let enums: Vec<String> = arr
                                .iter()
                                .filter_map(|v| v.as_str().map(|s| format!("\"{}\"", s)))
                                .collect();
                            if !enums.is_empty() {
                                return enums.join(" | ");
                            }
                        }
                    }
                    "string".to_string()
                }
                "number" => "number".to_string(),
                "integer" => "number".to_string(),
                "boolean" => "boolean".to_string(),
                "array" => {
                    if let Some(items) = schema.get("items") {
                        format!("{}[]", Self::json_schema_to_typescript(items, indent))
                    } else {
                        "Array<any>".to_string()
                    }
                }
                _ => "any".to_string(),
            }
        } else if let Some(one_of) = schema.get("oneOf") {
            // Defensive: already handled above, but just in case
            if let Some(arr) = one_of.as_array() {
                let mut out = String::new();
                let mut first = true;
                for variant in arr {
                    if !first {
                        out.push_str("\n | ");
                    } else {
                        first = false;
                    }
                    out.push_str(&Self::json_schema_to_typescript(variant, indent));
                }
                return out;
            }
            "any".to_string()
        } else {
            "any".to_string()
        }
    }

    /// Helper to template the tools section for system content rendering.
    fn template_tools_section(
        tools: &std::collections::BTreeMap<String, crate::chat::ToolNamespaceConfig>,
    ) -> String {
        let mut tool_sections = Vec::<String>::new();
        tool_sections.push("# Tools".to_string());
        for (_namespace, ns_config) in tools {
            let mut tool_section_content = Vec::<String>::new();
            tool_section_content.push(format!("## {}\n", ns_config.name));
            if let Some(desc) = &ns_config.description {
                for line in desc.lines() {
                    if !ns_config.tools.is_empty() {
                        tool_section_content.push(format!("// {}", line));
                    } else {
                        tool_section_content.push(line.to_string());
                    }
                }
            }
            if !ns_config.tools.is_empty() {
                tool_section_content.push(format!("namespace {} {{\n", ns_config.name));
                for tool in &ns_config.tools {
                    for line in tool.description.lines() {
                        tool_section_content.push(format!("// {}", line));
                    }
                    if let Some(params) = &tool.parameters {
                        let param_type = Self::json_schema_to_typescript(params, "");
                        tool_section_content.push(format!(
                            "type {} = (_: {}) => any;\n",
                            tool.name, param_type
                        ));
                    } else {
                        tool_section_content.push(format!("type {} = () => any;\n", tool.name));
                    }
                }
                tool_section_content.push(format!("}} // namespace {}", ns_config.name));
            }
            tool_sections.push(tool_section_content.join("\n"));
        }
        tool_sections.join("\n\n")
    }
}

trait Render<T: ?Sized> {
    fn render<B>(&self, item: &T, into: &mut B) -> anyhow::Result<()>
    where
        B: Extend<Rank>;
}

impl Render<Message> for HarmonyEncoding {
    fn render<B>(&self, message: &Message, into: &mut B) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        self.render_formatting_token_into(FormattingToken::Start, into)?;

        // render role then username
        if matches!(message.author.role, Role::Tool) {
            // for tools we only put the name
            if let Some(name) = &message.author.name {
                self.render_text_into(name, into)?;
            } else {
                anyhow::bail!("Tools should have a name!");
            }
        } else {
            // For users and assistants we put both the role, and optionally the user name.
            self.render_text_into(message.author.role.as_str(), into)?;
            if let Some(name) = &message.author.name {
                self.render_text_into(format!(":{}", name), into)?;
            }
        };

        // next render the header recipient, if there is one
        if let Some(recipient) = &message.recipient {
            if recipient != "all" {
                self.render_text_into(format!(" to={}", recipient), into)?;
            }
        }

        // next header channel
        if let Some(channel) = &message.channel {
            self.render_formatting_token_into(FormattingToken::Channel, into)?;
            self.render_text_into(channel, into)?;
        }

        // finally content type
        if let Some(content_type) = &message.content_type {
            self.render_text_into(format!(" {}", content_type), into)?;
        }

        self.render_formatting_token_into(FormattingToken::Message, into)?;
        for content in message.content.iter() {
            // SystemContent is only allowed in system messages
            if let crate::chat::Content::SystemContent(_) = content {
                anyhow::ensure!(
                    message.author.role == crate::chat::Role::System,
                    "SystemContent may only appear in system messages, found in {:?}",
                    message.author.role
                );
            }
            if let crate::chat::Content::DeveloperContent(_) = content {
                anyhow::ensure!(
                    message.author.role == crate::chat::Role::Developer,
                    "DeveloperContent may only appear in developer messages, found in {:?}",
                    message.author.role
                );
            }
            Render::<Content>::render(self, content, into)?;
        }

        // If there is a tool call we should render a tool call token
        if message.author.role == crate::chat::Role::Assistant && message.recipient.is_some() {
            self.render_formatting_token_into(FormattingToken::EndMessageAssistantToTool, into)?;
        } else {
            self.render_formatting_token_into(FormattingToken::EndMessage, into)?;
        }
        Ok(())
    }
}

// Dispatch Content variants to their specific Render implementations
impl Render<Content> for HarmonyEncoding {
    fn render<B>(&self, content: &Content, into: &mut B) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        match content {
            Content::Text(text) => Render::<TextContent>::render(self, text, into),
            Content::SystemContent(sys) => Render::<SystemContent>::render(self, sys, into),
            Content::DeveloperContent(dev) => {
                Render::<crate::chat::DeveloperContent>::render(self, dev, into)
            }
        }
    }
}

// Render plain text content
impl Render<TextContent> for HarmonyEncoding {
    fn render<B>(&self, text: &TextContent, into: &mut B) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        self.render_text_into(&text.text, into)
    }
}

// Render system-specific content (model identity, instructions, effort)
impl Render<SystemContent> for HarmonyEncoding {
    fn render<B>(&self, sys: &SystemContent, into: &mut B) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        let mut sections = Vec::<String>::new();

        let mut top_section = Vec::<String>::new();
        if let Some(model_id) = &sys.model_identity {
            top_section.push(model_id.clone());
        }
        if let Some(knowledge_cutoff) = &sys.knowledge_cutoff {
            top_section.push(format!("Knowledge cutoff: {knowledge_cutoff}"));
        }
        if let Some(conversation_start_date) = &sys.conversation_start_date {
            top_section.push(format!("Current date: {conversation_start_date}"));
        }
        if !top_section.is_empty() {
            sections.push(top_section.join("\n"));
        }

        let mut instructions_and_reasoning = Vec::<String>::new();
        if let Some(effort) = sys.reasoning_effort {
            let effort_str = match effort {
                ReasoningEffort::Low => "low",
                ReasoningEffort::Medium => "medium",
                ReasoningEffort::High => "high",
            };
            instructions_and_reasoning.push(format!("Reasoning: {effort_str}"));
        }
        if !instructions_and_reasoning.is_empty() {
            sections.push(instructions_and_reasoning.join("\n"));
        }

        if let Some(tools) = &sys.tools {
            if !tools.is_empty() {
                sections.push(Self::template_tools_section(tools));
            }
        }

        if let Some(channel_config) = &sys.channel_config {
            if !channel_config.valid_channels.is_empty() {
                let channels_str = channel_config.valid_channels.join(", ");
                let mut channels_header = format!("# Valid channels: {channels_str}.");
                if channel_config.channel_required {
                    channels_header.push_str(" Channel must be included for every message.");
                }
                if self.conversation_has_function_tools.load(Ordering::Relaxed) {
                    channels_header.push_str("\n");
                    channels_header.push_str(
                        "Calls to these tools must go to the commentary channel: 'functions'.",
                    );
                }
                sections.push(channels_header);
            }
        }
        let formatted = sections.join("\n\n");
        self.render_text_into(&formatted, into)?;
        Ok(())
    }
}

// Render developer-specific content (instructions, tools)
impl Render<crate::chat::DeveloperContent> for HarmonyEncoding {
    fn render<B>(&self, dev: &crate::chat::DeveloperContent, into: &mut B) -> anyhow::Result<()>
    where
        B: Extend<Rank>,
    {
        let mut sections = Vec::<String>::new();

        if let Some(instr) = &dev.instructions {
            sections.push("# Instructions".to_string());
            sections.push(instr.clone());
        }

        if let Some(tools) = &dev.tools {
            if !tools.is_empty() {
                sections.push(Self::template_tools_section(tools));
            }
        }
        let formatted = sections.join("\n\n");
        self.render_text_into(&formatted, into)?;
        Ok(())
    }
}

enum TakeUntilStatus {
    Found,
    EndOfStream,
}

impl TakeUntilStatus {
    fn was_found(&self) -> bool {
        matches!(self, TakeUntilStatus::Found)
    }
}

struct Parser<'a, I>
where
    I: Iterator<Item = Rank>,
{
    tokens: std::iter::Peekable<I>,
    encoding: &'a HarmonyEncoding,
}

impl<I> Parser<'_, I>
where
    I: Iterator<Item = Rank>,
{
    fn expect_special(&mut self, token: FormattingToken) -> anyhow::Result<Rank> {
        let next = self.tokens.next().context(format!(
            "Expected special token ({}), but out of tokens",
            token
        ))?;
        let expected = self.encoding.render_formatting_token(token)?;
        if next != expected {
            anyhow::bail!(
                "Expected special token ({}) {} but got {}",
                token,
                expected,
                next,
            );
        }
        Ok(next)
    }

    fn take_until_any(&mut self, ends: &HashSet<Rank>) -> (Vec<Rank>, TakeUntilStatus) {
        let mut out = vec![];
        for t in &mut self.tokens {
            if ends.contains(&t) {
                return (out, TakeUntilStatus::Found);
            }
            out.push(t);
        }
        (out, TakeUntilStatus::EndOfStream)
    }

    fn take_until(&mut self, end: Rank) -> (Vec<Rank>, TakeUntilStatus) {
        self.take_until_any(&HashSet::from([end]))
    }

    fn parse_header(&mut self, role: Option<Role>) -> anyhow::Result<ParsedHeader> {
        // FormattingToken::Message marks the end of the header.
        // Everything before that belongs to the header.
        let message_start_token = self
            .encoding
            .render_formatting_token(FormattingToken::Message)?;

        let (header_tokens, status) = self.take_until(message_start_token);
        if !status.was_found() {
            anyhow::bail!("Expected message start token but ran out of tokens");
        }

        // Decode the header into a UTF-8 string so we can reason about its structure.
        let mut header_string = self
            .encoding
            .tokenizer
            .decode_utf8(header_tokens)
            .context("could not decode header")?;

        // --------------------------------------------------------------------
        // 1. Extract the channel (if any)
        // --------------------------------------------------------------------
        // A channel, when present, is encoded as:
        //     <|channel|>CHANNEL_VALUE
        // where <|channel|> is the literal rendering of FormattingToken::Channel
        // and CHANNEL_VALUE is a contiguous string (no whitespace) naming the
        // channel.  The <|channel|> marker can appear before or after the
        // recipient part, but *always* before the optional content-type (which
        // must be last).
        let mut channel: Option<String> = None;
        if let Some(channel_marker) = self.encoding.mapped_format_token(FormattingToken::Channel) {
            if let Some(idx) = header_string.find(channel_marker) {
                // Slice parts around the marker
                let after_marker = &header_string[idx + channel_marker.len()..];

                // The channel value continues until the next ASCII whitespace,
                // the start of another special token ("<"), or end-of-string.
                let channel_end = after_marker
                    .find(|c: char| c.is_whitespace() || c == '<')
                    .unwrap_or(after_marker.len());
                let channel_value = &after_marker[..channel_end];
                if channel_value.is_empty() {
                    anyhow::bail!("channel marker present but no channel value found in header");
                }
                channel = Some(channel_value.to_string());

                // Remove the marker *and* the channel value from the header so
                // the remaining pieces can be parsed independently.
                let mut new_header = String::new();
                new_header.push_str(&header_string[..idx]);
                new_header.push_str(&after_marker[channel_end..]);
                header_string = new_header;
            }
        }

        // Trim extraneous whitespace that may have been introduced when we
        // removed the channel section.
        header_string = header_string.trim().to_string();

        // If the constrained format marker is present but not preceded by
        // whitespace (e.g. "to=foo<|constrain|>json"), insert a space before
        // the marker so that splitting on whitespace treats the content type
        // as a separate token.
        if let Some(constrain_marker) = self
            .encoding
            .mapped_format_token(FormattingToken::ConstrainedFormat)
        {
            if header_string.contains(constrain_marker) {
                header_string = header_string
                    .replace(constrain_marker, &format!(" {}", constrain_marker))
                    .trim()
                    .to_string();
            }
        }

        // --------------------------------------------------------------------
        // 2. Split the remaining header into whitespace-separated tokens.
        // --------------------------------------------------------------------
        // Debug output for development (only active when the `debug_header_parsing` cfg flag is
        // enabled).
        // For debugging purposes one might want to inspect the header string
        // at this point.  To avoid unwanted stdout noise in production use
        // the following (commented) line and recompile as needed.
        // println!("[DEBUG header] '{}'", header_string);

        let mut parts: Vec<&str> = header_string.split_ascii_whitespace().collect();

        // --------------------------------------------------------------------
        // 3. Determine the role (if not already provided).
        // --------------------------------------------------------------------
        let mut role_str_opt: Option<String> = None;
        let role = match role {
            Some(r) => r,
            None => {
                let role_str = parts
                    .first()
                    .context("message header did not contain a role")?;
                role_str_opt = Some((*role_str).to_string());
                let parsed_role = Role::try_from(*role_str);
                let out = match parsed_role {
                    Ok(r) => r,
                    Err(_) => {
                        // If recipient is present, treat as tool call
                        if parts.len() > 1 || (parts.len() == 1 && parts[0].starts_with("to=")) {
                            parts.remove(0); // Remove the unknown role string
                            Role::Tool
                        } else {
                            return Err(anyhow::anyhow!("Unknown role: {}", role_str));
                        }
                    }
                };
                out
            }
        };
        // If the role was supplied externally but also redundantly present in the
        // header itself, strip it off so that it does not interfere with the
        // parsing of the remaining fields.
        if let Some(first) = parts.first() {
            if *first == role.as_str() {
                parts.remove(0);
            }
        }

        // --------------------------------------------------------------------
        // 4. Identify recipient and content-type.
        // --------------------------------------------------------------------
        let mut recipient: Option<String> = None;
        let mut content_type: Option<String> = None;

        if !parts.is_empty() {
            // Determine whether the last token is a content-type or part of the
            // recipient specification.
            let num_tokens_before_pop = parts.len();
            let last_token_owned = parts.pop().unwrap().to_string();

            if last_token_owned.starts_with("to=") {
                // The header contains a recipient but *no* content-type.
                recipient = Some(last_token_owned.trim_start_matches("to=").to_string());
            } else if num_tokens_before_pop == 1 {
                // Only one token total (after potential role removal) and it doesn't start
                // with "to=" => interpret it as a standalone recipient.
                recipient = Some(last_token_owned);
            } else {
                // More than one token and the last one is not a recipient -> treat as content-type.
                content_type = Some(last_token_owned);

                // After removing the content-type there may be exactly one token describing the recipient.
                if !parts.is_empty() {
                    if parts.len() != 1 {
                        anyhow::bail!("Could not parse header: too many tokens remaining after extracting content-type and recipient");
                    }
                    let raw_recipient = parts.pop().unwrap();
                    recipient = if let Some(stripped) = raw_recipient.strip_prefix("to=") {
                        Some(stripped.to_string())
                    } else {
                        Some(raw_recipient.to_string())
                    };
                }
            }
        }

        // After processing, no unparsed tokens should remain.
        anyhow::ensure!(
            parts.is_empty(),
            "unexpected tokens remaining in message header: {:?}",
            parts
        );
        // We have successfully parsed the header.
        let author = if role == Role::Tool {
            let name = role_str_opt;
            Author { role, name }
        } else {
            Author { role, name: None }
        };
        Ok(ParsedHeader {
            author,
            recipient,
            channel,
            content_type,
        })
    }

    /// Parses a message from the remaining tokens.
    ///
    /// Returns the message and a boolean indicating whether end of stream was reached.
    fn parse_message(&mut self, role: Option<Role>) -> anyhow::Result<(Message, bool)> {
        let start_token = self
            .encoding
            .render_formatting_token(FormattingToken::Start)?;
        match role {
            Some(_) => {
                if let Some(&next) = self.tokens.peek() {
                    if next == start_token {
                        self.tokens.next();
                    }
                } else {
                    anyhow::bail!("Expected at least one token while parsing message");
                }
            }
            None => {
                self.expect_special(FormattingToken::Start)?;
            }
        }
        let header = self.parse_header(role)?;
        let ParsedHeader {
            author,
            recipient,
            channel,
            content_type,
        } = header;

        // TODO other content types
        // since we bail on anything other than just the role in the header for now, we can assume
        // that the content type is text
        let end_tokens = self.encoding.stop_tokens()?;
        let (remaining_tokens, status) = self.take_until_any(&end_tokens);
        let remaining_text = self
            .encoding
            .tokenizer
            .decode_utf8(remaining_tokens)
            .context("could not decode message content")?;
        let did_reach_end_of_stream = match status {
            TakeUntilStatus::Found => self.tokens.peek().is_none(),
            TakeUntilStatus::EndOfStream => true,
        };
        Ok((
            Message {
                author,
                content: vec![Content::Text(TextContent {
                    text: remaining_text,
                })],
                channel,
                recipient,
                content_type,
            },
            did_reach_end_of_stream,
        ))
    }
}

// ---------------------------------------------------------------------------
// Streamable parsing ---------------------------------------------------------
// ---------------------------------------------------------------------------

/// Incremental parser that can consume tokens one by one.
///
/// It keeps track of all tokens seen so far, exposes all fully parsed messages
/// and retains the partially parsed state of the current message.
pub struct StreamableParser {
    encoding: HarmonyEncoding,
    next_role: Option<Role>,
    tokens: Vec<Rank>,
    messages: Vec<Message>,
    state: StreamState,
    stop_tokens: HashSet<Rank>,
    last_content_delta: Option<String>,
    undecoded_tokens: Vec<Rank>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum StreamState {
    ExpectStart,
    Header {
        header_tokens: Vec<Rank>,
    },
    Content {
        header: ParsedHeader,
        content_tokens: Vec<Rank>,
    },
}

impl StreamableParser {
    /// Create a new streaming parser starting with the given role.
    pub fn new(encoding: HarmonyEncoding, role: Option<Role>) -> anyhow::Result<Self> {
        let stop_tokens = encoding.stop_tokens()?;
        let (state, next_role) = match role {
            Some(role) => (
                StreamState::Header {
                    header_tokens: Vec::new(),
                },
                Some(role),
            ),
            None => (StreamState::ExpectStart, None),
        };
        Ok(Self {
            encoding,
            next_role,
            tokens: Vec::new(),
            messages: Vec::new(),
            state,
            stop_tokens,
            last_content_delta: None,
            undecoded_tokens: Vec::new(),
        })
    }

    /// Consume a single token and update the internal state.
    pub fn process(&mut self, token: Rank) -> anyhow::Result<&mut Self> {
        self.tokens.push(token);
        // Clone next_role up front to avoid borrow checker issues
        let next_role_clone = self.next_role.clone();
        match &mut self.state {
            StreamState::ExpectStart => {
                let start = self
                    .encoding
                    .render_formatting_token(FormattingToken::Start)?;
                if token == start {
                    self.state = StreamState::Header {
                        header_tokens: Vec::new(),
                    };
                } else {
                    anyhow::bail!(
                        "Unexpected token {} while expecting start token {}",
                        token,
                        start
                    );
                }
            }
            StreamState::Header { header_tokens } => {
                let msg_tok = self
                    .encoding
                    .render_formatting_token(FormattingToken::Message)?;
                if token == msg_tok {
                    // Clone the tokens and next_role, then clear the state before parsing
                    let header_tokens_cloned = header_tokens.clone();
                    let next_role_cloned = next_role_clone;
                    // Set state to dummy to drop mutable borrow
                    self.state = StreamState::ExpectStart;
                    let header =
                        self.parse_header_from_tokens(&header_tokens_cloned, next_role_cloned)?;
                    self.next_role = None;
                    self.state = StreamState::Content {
                        header,
                        content_tokens: Vec::new(),
                    };
                } else {
                    header_tokens.push(token);
                }
            }
            StreamState::Content {
                header,
                content_tokens,
            } => {
                if self.stop_tokens.contains(&token) {
                    let text = self.encoding.tokenizer().decode_utf8(content_tokens)?;
                    let message = Message {
                        author: header.author.clone(),
                        recipient: header.recipient.clone(),
                        channel: header.channel.clone(),
                        content_type: header.content_type.clone(),
                        content: vec![Content::Text(TextContent { text })],
                    };
                    self.messages.push(message);
                    self.state = StreamState::ExpectStart;
                    self.last_content_delta = None;
                    self.undecoded_tokens.clear();
                } else {
                    self.undecoded_tokens.push(token);
                    // some tokens might not appropriately decode on their own. If they don't
                    // we will collect them until they eventually decode
                    match self
                        .encoding
                        .tokenizer()
                        .decode_utf8(&self.undecoded_tokens)
                    {
                        Ok(decoded) => {
                            content_tokens.extend(self.undecoded_tokens.iter().copied());
                            self.last_content_delta = Some(decoded);
                            self.undecoded_tokens.clear();
                        }
                        Err(_) => {
                            self.last_content_delta = None;
                        }
                    }
                }
            }
        }
        Ok(self)
    }

    fn parse_header_from_tokens(
        &self,
        header_tokens: &[Rank],
        role: Option<Role>,
    ) -> anyhow::Result<ParsedHeader> {
        let mut header_string = self
            .encoding
            .tokenizer()
            .decode_utf8(header_tokens)
            .context("could not decode header")?;

        let mut channel: Option<String> = None;
        if let Some(channel_marker) = self.encoding.mapped_format_token(FormattingToken::Channel) {
            if let Some(idx) = header_string.find(channel_marker) {
                let after_marker = &header_string[idx + channel_marker.len()..];
                let channel_end = after_marker
                    .find(|c: char| c.is_whitespace() || c == '<')
                    .unwrap_or(after_marker.len());
                let channel_value = &after_marker[..channel_end];
                if channel_value.is_empty() {
                    anyhow::bail!("channel marker present but no channel value found in header");
                }
                channel = Some(channel_value.to_string());

                let mut new_header = String::new();
                new_header.push_str(&header_string[..idx]);
                new_header.push_str(&after_marker[channel_end..]);
                header_string = new_header;
            }
        }

        // Trim extraneous whitespace that may have been introduced when we
        // removed the channel section.
        header_string = header_string.trim().to_string();

        // If the constrained format marker is present but not preceded by
        // whitespace (e.g. "to=foo<|constrain|>json"), insert a space before
        // the marker so that splitting on whitespace treats the content type
        // as a separate token.
        if let Some(constrain_marker) = self
            .encoding
            .mapped_format_token(FormattingToken::ConstrainedFormat)
        {
            if header_string.contains(constrain_marker) {
                header_string = header_string
                    .replace(constrain_marker, &format!(" {constrain_marker}"))
                    .trim()
                    .to_string();
            }
        }

        let mut parts: Vec<&str> = header_string.split_ascii_whitespace().collect();

        let mut role_str_opt: Option<String> = None;
        let role = match role {
            Some(r) => r,
            None => {
                let role_str = parts
                    .first()
                    .context("message header did not contain a role")?;
                role_str_opt = Some((*role_str).to_string());
                let parsed_role = Role::try_from(*role_str);
                let out = match parsed_role {
                    Ok(r) => r,
                    Err(_) => {
                        // If recipient is present, treat as tool call
                        if parts.len() > 1 || (parts.len() == 1 && parts[0].starts_with("to=")) {
                            parts.remove(0); // Remove the unknown role string
                            Role::Tool
                        } else {
                            return Err(anyhow::anyhow!("Unknown role: {}", role_str));
                        }
                    }
                };
                out
            }
        };

        if let Some(first) = parts.first() {
            if *first == role.as_str() {
                parts.remove(0);
            }
        }

        let mut recipient: Option<String> = None;
        let mut content_type: Option<String> = None;

        if !parts.is_empty() {
            let num_tokens_before_pop = parts.len();
            let last_token_owned = parts.pop().unwrap().to_string();

            if last_token_owned.starts_with("to=") {
                recipient = Some(last_token_owned.trim_start_matches("to=").to_string());
            } else if num_tokens_before_pop == 1 {
                recipient = Some(last_token_owned);
            } else {
                content_type = Some(last_token_owned);

                if !parts.is_empty() {
                    if parts.len() != 1 {
                        anyhow::bail!(
                            "Could not parse header: too many tokens remaining after extracting content-type and recipient"
                        );
                    }
                    let raw_recipient = parts.pop().unwrap();
                    recipient = if let Some(stripped) = raw_recipient.strip_prefix("to=") {
                        Some(stripped.to_string())
                    } else {
                        Some(raw_recipient.to_string())
                    };
                }
            }
        }

        anyhow::ensure!(
            parts.is_empty(),
            "unexpected tokens remaining in message header: {:?}",
            parts
        );
        let author = if role == Role::Tool {
            let name = role_str_opt;
            Author { role, name }
        } else {
            Author { role, name: None }
        };
        Ok(ParsedHeader {
            author,
            recipient,
            channel,
            content_type,
        })
    }

    /// Return the textual content of the current message so far.
    pub fn current_content(&self) -> anyhow::Result<String> {
        match &self.state {
            StreamState::Content { content_tokens, .. } => self
                .encoding
                .tokenizer()
                .decode_utf8(content_tokens)
                .map_err(|e| anyhow::anyhow!(e)),
            _ => Ok(String::new()),
        }
    }

    /// Role of the current message if it has been parsed.
    pub fn current_role(&self) -> Option<Role> {
        match &self.state {
            StreamState::Content { header, .. } => Some(header.author.role.clone()),
            _ => self.next_role.clone(),
        }
    }

    /// Current content type if known.
    pub fn current_content_type(&self) -> Option<String> {
        match &self.state {
            StreamState::Content { header, .. } => header.content_type.clone(),
            _ => None,
        }
    }

    /// Decode the last content delta if available.
    pub fn last_content_delta(&self) -> anyhow::Result<Option<String>> {
        Ok(self.last_content_delta.clone())
    }

    /// All fully parsed messages so far.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// All tokens that were fed into the parser.
    pub fn tokens(&self) -> &[Rank] {
        &self.tokens
    }

    /// Expose the current state as a JSON string for Python interop.
    pub fn state_json(&self) -> anyhow::Result<String> {
        #[derive(serde::Serialize)]
        #[serde(tag = "state")]
        enum SerializableStreamState<'a> {
            ExpectStart,
            Header {
                header_tokens: &'a [Rank],
            },
            Content {
                header: &'a ParsedHeader,
                content_tokens: &'a [Rank],
            },
        }
        let serializable = match &self.state {
            StreamState::ExpectStart => SerializableStreamState::ExpectStart,
            StreamState::Header { header_tokens } => {
                SerializableStreamState::Header { header_tokens }
            }
            StreamState::Content {
                header,
                content_tokens,
            } => SerializableStreamState::Content {
                header,
                content_tokens,
            },
        };
        Ok(serde_json::to_string(&serializable)?)
    }

    /// Return the current recipient if known.
    pub fn current_recipient(&self) -> Option<String> {
        match &self.state {
            StreamState::Content { header, .. } => header.recipient.clone(),
            _ => None,
        }
    }

    /// Return the current channel if known.
    pub fn current_channel(&self) -> Option<String> {
        match &self.state {
            StreamState::Content { header, .. } => header.channel.clone(),
            _ => None,
        }
    }
}

// Add config struct for rendering
#[derive(Clone, Debug)]
pub struct RenderConversationConfig {
    pub auto_drop_analysis: bool,
}

impl Default for RenderConversationConfig {
    fn default() -> Self {
        Self {
            auto_drop_analysis: true,
        }
    }
}
