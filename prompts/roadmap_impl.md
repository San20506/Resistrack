# Module Implementation Prompt

Purpose: Template prompt for implementing a specific module from the roadmap.

## Template

You are a Senior Software Engineer. Your task is to implement the following module for the ResisTrack project.

### Module Details
*   **Module ID:** {MODULE_ID}
*   **Title:** {MODULE_TITLE}

### Specification
{SPEC_CONTENT}

### Dependencies
The following upstream modules have been completed and their outputs are available:
{DEPENDENCIES}

### Constraints & Rules
All implementation must strictly follow the project's Agent Rules:
{AGENT_RULES}

### Instructions
1.  **Read Spec:** Thoroughly review the module specification, including scope and requirements.
2.  **Verify Dependencies:** Ensure all required dependency outputs are available and understood.
3.  **Implement Logic:** Write the implementation code in the `impl/` directory of the module.
4.  **Write Tests:** Create comprehensive tests in the `tests/` directory based on the "done-when" checklist in the spec.
5.  **Compliance Check:** Verify that the implementation complies with all Agent Rules (e.g., security, privacy, architectural patterns).
6.  **Output:** Provide the final implementation files and test files.

Ensure the code is clean, documented, and follows the project's coding standards.
