# Learnings

## React + Vite Setup
- When setting up a new Vite project manually (without `create-vite`), ensure `vite-env.d.ts` includes `/// <reference types="vite/client" />` to avoid TypeScript errors with `import.meta.env`.
- The `tsconfig.json` needs `moduleResolution: "bundler"` for modern Vite setups.
- Tailwind CSS setup with Vite requires `postcss.config.js` and `tailwind.config.js`.

## TypeScript Configuration
- Strict mode is essential for robust type checking.
- Path aliases (e.g., `@/*`) in `tsconfig.json` must be mirrored in `vite.config.ts` using `resolve.alias`.

## Project Structure
- `src/types/index.ts` is a good place for shared type definitions, especially when mirroring backend schemas.
- `src/api/client.ts` with a configured Axios instance and stub functions allows for frontend development to proceed before the backend is fully ready.
