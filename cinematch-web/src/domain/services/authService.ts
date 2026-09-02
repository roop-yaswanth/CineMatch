/**
 * AuthService — domain service wrapping authentication operations.
 */

import type { AuthRepository } from "@/domain/repositories/AuthRepository";
import type { UserSession } from "@/domain/types/movie";
import { authRepository } from "@/data/repositories/HttpAuthRepository";

// The singleton uses the concrete adapter; callers import the service, not fetch.
const repo: AuthRepository = authRepository;

export async function signInWithGoogle(credential: string): Promise<UserSession> {
  return repo.loginWithGoogle(credential);
}

export async function refreshSession(authToken: string): Promise<UserSession> {
  return repo.refreshSession(authToken);
}

export async function signOut(sessionId: string): Promise<void> {
  return repo.logout(sessionId);
}
