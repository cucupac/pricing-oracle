export interface HealthStatusResponse {
  status: "healthy" | "degraded" | "unhealthy";
  timestamp: string;
}
