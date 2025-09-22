export class ExternalServiceError extends Error {
  constructor(message: string, public readonly cause?: unknown, public readonly status?: number) {
    super(message);
    this.name = "ExternalServiceError";
  }
}
