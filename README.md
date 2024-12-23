
#### 📨 PromptCLI - AI-powered Command Generation

## 📖 Overview

PromptCLI is a powerful tool that transforms natural language inputs into executable terminal commands, utilizing GPT-based language models. This AI-powered interface is designed to understand and generate accurate, multi-step shell commands for various operating systems (Windows, Mac, Linux). The application ensures security and accuracy by validating the generated commands and implementing safety measures. Built with Streamlit, PromptCLI provides an easy-to-use interface for real-time command generation with contextual understanding.

## 💡 Problem Statement

Manually creating terminal commands can be difficult, especially when the required commands are complex or involve multiple steps. Additionally, the risk of executing harmful or incorrect commands is high. PromptCLI bridges this gap by converting user-friendly natural language into correct and safe terminal commands, while supporting various platforms and reducing errors.

## ✨ Features

- 🔍 GPT-Powered Command Generation: Utilizes GPT-based models to translate natural language prompts into executable shell commands.

- 🖥️ Cross-Platform Compatibility: Supports generating commands for Windows, Linux, and macOS, tailored to the unique syntax and requirements of each platform.

- 🔒 Command Safety Checks: Integrated safety checks and guardrails ensure that generated commands are safe to execute, preventing security risks and harmful operations.

- 🔍 OS-Specific Validation: Performs command validation and syntax checking for each operating system, ensuring accuracy and minimizing errors by 40%.

- 📊 Real-Time Interaction: The user-friendly Streamlit interface generates commands instantly based on user input.

- 🚀 Advanced Contextual Understanding: Handles complex, multi-step operations and ensures commands are appropriately sequenced.

## 💻 Requirements

🐍 Python 3.x
📦 Streamlit
📦 OpenAI API
📦 LangChain
📦 Shell Scripting

## ▶️ How to Run

- 📥 Clone this repository:
git clone https://github.com/prachirathi07/promptcli.git
cd promptcli


- 🏃 Run the application:
streamlit run main.py

## 🔍 Key Features

- Natural Language to Shell Command Conversion: Easily convert spoken or written instructions into complex shell commands.

- Cross-Platform Support: Supports command generation tailored for different operating systems (Windows, Linux, macOS).

- Safety and Security: Built-in safety measures to prevent the execution of harmful commands, ensuring a secure user experience.

- OS-Specific Command Validation: Ensures commands are syntactically correct for each platform, minimizing errors and improving command accuracy.

## 🛠️ Technologies Used

- OpenAI API: GPT-powered language model for converting natural language to terminal commands.
- Python: Backend logic and command processing.
- Streamlit: Web framework for creating the interactive user interface.
- LangChain: For chaining language model tasks and ensuring smooth operation across complex multi-step processes.
- Shell Scripting: For handling terminal commands across different operating systems.
## 🔮 Future Enhancements

- Voice Input Support: Enable voice command functionality to generate commands by simply speaking to the app.

- Command Execution: Allow users to execute the generated commands directly from the interface for enhanced productivity.

- Advanced Customization: Add features to allow users to specify detailed command parameters, options, and flags.

## 👤 Author

Prachi Rathi

