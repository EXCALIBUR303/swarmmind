import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EmptyState } from "./EmptyState";

describe("EmptyState", () => {
  it("renders the title and detail text", () => {
    render(<EmptyState title="Swarm Lab" detail="coming soon" />);
    expect(screen.getByText("Swarm Lab")).toBeInTheDocument();
    expect(screen.getByText("coming soon")).toBeInTheDocument();
  });

  it("renders without detail", () => {
    render(<EmptyState title="Settings" />);
    expect(screen.getByText("Settings")).toBeInTheDocument();
  });
});
