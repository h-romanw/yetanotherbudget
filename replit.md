# Yet Another Budget - AI-Powered Spending Analysis

## Overview

Yet Another Budget is a personal finance application that analyzes spending patterns using AI. The application allows users to import financial data, view transactions, and understand their spending habits through automated categorization and analysis. The platform is built as a full-stack TypeScript application with a React frontend and Express backend.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack**
- React with TypeScript for type safety and component-based UI
- Vite as the build tool and development server for fast hot module replacement
- Wouter for lightweight client-side routing
- TanStack Query (React Query) for server state management and caching

**UI Framework Decision**
- shadcn/ui component library built on Radix UI primitives
- Tailwind CSS for utility-first styling with custom design tokens
- Rationale: Provides accessible, customizable components while maintaining design consistency through CSS variables

**State Management Approach**
- React Query handles server state with aggressive caching (staleTime: Infinity)
- Local component state via React hooks for UI-specific state
- Global query client configured to avoid unnecessary refetches
- Custom query functions with 401 error handling for authentication flows

**Design System**
- Custom CSS variables for typography (Manrope font family)
- Neutral color scheme as base with HSL color system
- Responsive breakpoints with mobile-first approach (768px breakpoint)
- Custom shadow system including "bold-projection" for depth

### Backend Architecture

**Server Framework**
- Express.js for HTTP server and middleware pipeline
- Node.js with ESM modules for modern JavaScript features
- Separate build process using esbuild for production bundling

**API Structure**
- RESTful API convention with `/api` prefix for all endpoints
- Middleware-based request logging with duration tracking
- JSON request/response bodies with automatic parsing
- Centralized error handling middleware

**Development vs Production**
- Vite middleware integration in development for HMR
- Static file serving in production from built assets
- Environment-based configuration (NODE_ENV)
- Replit-specific plugins for development tooling

### Data Storage Solutions

**Database Architecture**
- PostgreSQL as the primary relational database
- Neon serverless PostgreSQL for cloud deployment
- Connection pooling via @neondatabase/serverless driver

**ORM and Schema Management**
- Drizzle ORM for type-safe database queries
- Schema-first approach with TypeScript types inferred from schema
- Drizzle Kit for migrations with schema in `shared/schema.ts`
- Zod integration via drizzle-zod for runtime validation

**Storage Abstraction Layer**
- IStorage interface for repository pattern implementation
- MemStorage in-memory implementation for development/testing
- Designed to swap implementations without changing application logic
- Currently implements user management (getUser, getUserByUsername, createUser)

**Data Model**
- Users table with UUID primary keys (auto-generated)
- Username/password authentication fields
- Shared schema between client and server via `shared/` directory
- Type inference ensures client-server type consistency

### Authentication and Authorization

**Current Implementation**
- Basic user schema with username/password fields
- Session management preparation via connect-pg-simple (PostgreSQL session store)
- Cookie-based sessions anticipated (session store configured)

**Security Considerations**
- Password storage field exists but hashing implementation not visible
- Session configuration for production deployment ready
- 401 error handling in query client for unauthorized access

### Module Resolution and Project Structure

**TypeScript Path Aliases**
- `@/*` maps to client source directory
- `@shared/*` for shared types and schemas
- `@assets/*` for static assets
- Enables clean imports and prevents relative path complexity

**Monorepo Structure**
- `/client` - Frontend React application
- `/server` - Backend Express application  
- `/shared` - Shared TypeScript schemas and types
- `/migrations` - Database migration files
- Separate tsconfig with unified compilation settings

**Build Pipeline**
- Client: Vite builds to `dist/public`
- Server: esbuild bundles to `dist/index.js`
- Production serves client build as static files
- Development runs concurrent dev servers

## External Dependencies

### Third-Party UI Libraries
- Radix UI component primitives (20+ components including dialogs, dropdowns, tooltips, etc.)
- Embla Carousel for carousel functionality
- cmdk for command palette interfaces
- Lucide React for icon system
- class-variance-authority for component variant management

### Data Fetching and Forms
- @tanstack/react-query for server state management
- @hookform/resolvers for form validation
- react-hook-form (peer dependency) for form handling
- Zod for schema validation

### Database and ORM
- Drizzle ORM (v0.39.1) for database queries
- @neondatabase/serverless for PostgreSQL connection
- drizzle-kit for migrations and schema management
- connect-pg-simple for PostgreSQL session storage

### Styling and Design
- Tailwind CSS with PostCSS processing
- Autoprefixer for CSS vendor prefixes
- Custom font loading (Manrope, DM Sans, Fira Code, Geist Mono, Architects Daughter)

### Development Tools
- @replit/vite-plugin-runtime-error-modal for error overlays
- @replit/vite-plugin-cartographer for code mapping
- @replit/vite-plugin-dev-banner for development indicators
- tsx for TypeScript execution in development
- esbuild for production bundling

### Utility Libraries
- date-fns for date manipulation
- clsx and tailwind-merge (via cn utility) for class name management
- nanoid for unique ID generation
- vaul for drawer components