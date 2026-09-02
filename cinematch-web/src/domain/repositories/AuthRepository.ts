/**
 * AuthRepository — abstraction for authentication.
 * Domain (login, session refresh, logout) depends on this interface.
 * Whether the backend is Firebase, Auth0, or custom SQL is hidden.
 */

import type { UserSession } from "../types/movie";

export interface AuthRepository {
  loginWithGoogle(credential: string): Promise<UserSession>;
  refreshSession(authToken: string): Promise<UserSession>;
  logout(sessionId: string): Promise<void>;
}

export interface SessionStore {
  getSession(): UserSession | null;
  saveSession(session: UserSession): void;
  clearSession(): void;
  getAuthToken(): string | null;
  saveAuthToken(token: string): void;
  clearAuthToken(): void;
}
