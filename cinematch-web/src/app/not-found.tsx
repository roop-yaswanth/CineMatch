// Catch-all 404. Next renders this whenever a route doesn't match — including
// when notFound() is called from a server component inside an existing route.

import ErrorView from "@/components/ui/ErrorView";

export default function NotFound() {
  return <ErrorView code={404} />;
}
