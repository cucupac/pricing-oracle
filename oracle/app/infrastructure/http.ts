import axios, { AxiosInstance } from "axios";
import { Env } from "../entrypoints/api/env";
import { ExternalServiceError } from "../core/errors";

export function makeHttpClient(baseURL: string): AxiosInstance {
  const client = axios.create({
    baseURL,
    timeout: Env.HTTP_TIMEOUT_MS,
    headers: { "Content-Type": "application/json" }
  });

  client.interceptors.response.use(undefined, async (error) => {
    const cfg = error.config;
    const status = error.response?.status;
    const retriable = !status || (status >= 500 && status < 600);
    (cfg as any).__retryCount = (cfg as any).__retryCount || 0;
    if (retriable && (cfg as any).__retryCount < Env.HTTP_RETRIES) {
      (cfg as any).__retryCount++;
      return client(cfg);
    }
    throw new ExternalServiceError(
      `HTTP ${status ?? "ERR"} for ${cfg?.baseURL ?? ""}${cfg?.url ?? ""}`,
      error,
      status
    );
  });

  return client;
}
