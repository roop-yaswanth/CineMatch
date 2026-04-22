import type { UserSession } from "@/lib/api";

type SessionRouteInfo = Pick<UserSession, "onboarding_complete">;

export function sessionHomePath(session: SessionRouteInfo): "/dashboard" | "/onboarding" {
  return session.onboarding_complete ? "/dashboard" : "/onboarding";
}

