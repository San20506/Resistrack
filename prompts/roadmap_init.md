# Roadmap Initialization Prompt

Purpose: Template prompt for initializing a new project roadmap from requirements documents.

## Template

You are an expert Technical Architect. Your task is to initialize a phased project roadmap for {PROJECT_NAME}.

### Project Context
*   **Goal:** {GOAL}
*   **Stack:** {STACK}
*   **Constraints:** {CONSTRAINTS}
*   **Timeline:** {TIMELINE}

### Source Documents
Please analyze the following provided documents:
1. PRD
2. Technical Architecture
3. Agent Rules

### Instructions
1.  **Analyze Documents:** Review the provided context and source documents to understand the full scope of the project.
2.  **Identify Phases:** Organize the work into logical phases based on dependency order (e.g., Foundation, Core Engine, Integration, UX, Hardening).
3.  **Modular Breakdown:** Break each phase into independent modules. Each module should represent approximately 1 to 3 days of work.
4.  **Define Dependencies:** Clearly map dependencies between modules, ensuring a logical build order.
5.  **Success Criteria:** Define specific success criteria for each phase.
6.  **Output Format:** Generate the complete roadmap in Markdown format.

Each module entry must include:
*   Module ID (e.g., m1_1)
*   Title
*   Brief description
*   Dependencies
*   Expected outcomes
