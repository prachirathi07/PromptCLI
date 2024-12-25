import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, List, Optional
from dataclasses import dataclass
import re


@dataclass
class CommandResponse:
    command: str
    explanation: str
    safety_warnings: List[str]
    os_specific_notes: str


class PromptCLI:
    def __init__(self, api_key: str):
        """Initialize PromptCLI with LangChain GroqAPI components."""
        self.groq_model = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-70b-versatile"  # Adjust model name as necessary
        )
        self.safety_patterns = self.load_safety_patterns()

    def generate_command(self, task: str, os_type: str) -> Optional[CommandResponse]:
        """Generate and validate a command using LangChain GroqAPI."""
        try:
            # Prepare messages for generating command (no emojis used in the message content)
            messages = [
                SystemMessage(content=f"You are an expert CLI command generator for {os_type}."),
                HumanMessage(content=f"Convert the following natural language description into an accurate terminal command: {task}.")
            ]

            # Generate command
            response = self.groq_model.invoke(messages)
            command = response.content.strip()  # Adjust based on actual response structure

            # Remove any non-ASCII characters (including emojis)
            command = self.remove_emojis(command)

            # Prepare messages for explanation
            explanation_messages = [
                SystemMessage(content="Analyze the given command."),
                HumanMessage(content=f"Explain what this command does: {command}. Include any potential risks or considerations.")
            ]

            # Generate explanation
            explanation_response = self.groq_model.invoke(explanation_messages)
            explanation = explanation_response.content.strip()  # Adjust based on actual response structure

            # Remove any non-ASCII characters (including emojis) from the explanation
            explanation = self.remove_emojis(explanation)

            # Validate command safety
            safety_warnings = self.validate_command_safety(command)

            return CommandResponse(
                command=command,
                explanation=explanation,
                safety_warnings=safety_warnings,
                os_specific_notes=f"Command generated for {os_type} environment."
            )

        except Exception as e:
            st.error(f"Error generating command: {str(e)}")
            return None

    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove any emojis or non-ASCII characters from the text."""
        # Regex pattern to match all non-ASCII characters (including emojis)
        return re.sub(r'[^\x00-\x7F]+', '', text)

    @staticmethod
    def load_safety_patterns() -> Dict[str, List[str]]:
        """Load patterns for command safety checking."""
        return {
            "dangerous_commands": [
                "rm -rf", "del /F /S /Q", "format",
                "mkfs", ">", "dd if="
            ],
            "sensitive_paths": [
                "/etc/passwd", "C:\\Windows\\System32",
                "/dev/sd", "C:\\Program Files"
            ]
        }

    def validate_command_safety(self, command: str) -> List[str]:
        """Check command against safety patterns."""
        warnings = []

        for pattern in self.safety_patterns["dangerous_commands"]:
            if pattern in command.lower():
                warnings.append(f"⚠️ Warning: Command contains potentially dangerous operation: {pattern}")

        for path in self.safety_patterns["sensitive_paths"]:
            if path in command:
                warnings.append(f"⚠️ Warning: Command attempts to access sensitive path: {path}")

        return warnings


def main():
    st.set_page_config(
        page_title="PromptCLI - AI Command Generator",
        page_icon="💻",  # Added emoji to page icon
        layout="wide"
    )

    st.title("PromptCLI - AI Command Generator 👨‍💻💻")  # Added emojis to title for better user engagement

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []

    # API key input
    api_key = st.sidebar.text_input("Enter Your GroqAPI Key 🔑", type="password")

    if not api_key:
        st.warning("Please enter your GroqAPI key to continue. 🔑")
        return

    # Initialize PromptCLI with GroqAPI
    cli = PromptCLI(api_key)

    # OS Selection
    os_options = {
        "Windows": "Windows",
        "Darwin": "MacOS",
        "Linux": "Linux"
    }
    selected_os = st.selectbox(
        "Select Operating System 🖥️",
        options=list(os_options.values())
    )

    # Task input
    task = st.text_area("Describe what you want to do 📝:", 
                        placeholder="E.g., 'list all files in current directory'")

    if st.button("Generate Command ⚡"):
        if task:
            with st.spinner("Generating command... ⏳"):
                result = cli.generate_command(task, selected_os)

                if result:
                    # Display command with copy button
                    st.code(result.command, language="bash")

                    # Display explanation
                    with st.expander("Command Explanation 🧑‍🏫"):
                        st.write(result.explanation)

                    # Display safety warnings if any
                    if result.safety_warnings:
                        st.warning("\n".join(result.safety_warnings))

                    # Add to history
                    st.session_state.history.append({
                        "task": task,
                        "command": result.command,
                        "os": selected_os
                    })

    # Display command history
    if st.session_state.history:
        with st.expander("Command History 📜"):
            for idx, item in enumerate(reversed(st.session_state.history)):
                st.text(f"Task: {item['task']}")
                st.code(item['command'], language="bash")
                st.text(f"OS: {item['os']}")
                st.divider()

    # Footer with some style
    st.markdown(
        "<br><p style='text-align: center; font-size: 14px;'>Made with ❤️ by Prachi Rathi</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
