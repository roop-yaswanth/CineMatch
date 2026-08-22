"use client";

import ErrorView from "@/components/ui/ErrorView";

export default function ServerErrorPage() {
  return (
    <ErrorView
      code={500}
      title="Server is waking up"
      description="The server enters sleep mode due to limited Hugging Face free space. Signing in automatically triggers a server restart which takes about 2 minutes. Please retry after 2 minutes."
      showTimer={true}
      action={{
        label: "Retry Login",
        onClick: () => {
          if (typeof window !== "undefined") {
            window.location.href = "/login";
          }
        },
      }}
    />
  );
}
