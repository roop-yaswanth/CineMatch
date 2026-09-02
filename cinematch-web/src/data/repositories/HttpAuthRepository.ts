import { httpRequest } from "@/infrastructure/http/HttpClient";
import type { AuthRepository } from "@/domain/repositories/AuthRepository";
import type { UserSession } from "@/domain/types/movie";

export class HttpAuthRepository implements AuthRepository {
  loginWithGoogle(credential: string): Promise<UserSession> {
    return httpRequest<UserSession>("/api/auth/google", {
      method: "POST",
      body: JSON.stringify({ credential }),
    });
  }
  refreshSession(authToken: string): Promise<UserSession> {
    return httpRequest<UserSession>("/api/auth/refresh", {
      method: "POST",
      body: JSON.stringify({ auth_token: authToken }),
    });
  }
  logout(sessionId: string): Promise<void> {
    return httpRequest<void>("/api/logout", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    });
  }
}

export const authRepository = new HttpAuthRepository();
