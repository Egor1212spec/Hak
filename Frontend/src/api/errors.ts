import axios from "axios";

const parseDetail = async (data: unknown): Promise<string> => {
  if (!data) return "";

  if (data instanceof Blob) {
    const text = await data.text();
    if (!text) return "";
    try {
      const parsed = JSON.parse(text) as { detail?: string };
      if (parsed && typeof parsed === "object" && "detail" in parsed) {
        return String(parsed.detail ?? "");
      }
      return text;
    } catch {
      return text;
    }
  }

  if (typeof data === "string") {
    return data;
  }

  if (typeof data === "object" && data && "detail" in data) {
    return String((data as { detail?: string }).detail ?? "");
  }

  return "";
};

export const getApiErrorMessage = async (
  err: unknown,
  fallback: string
): Promise<string> => {
  if (!axios.isAxiosError(err)) return fallback;

  const status = err.response?.status ?? "unknown";
  if (!err.response) return fallback;

  const detail = await parseDetail(err.response.data);
  return `Ошибка сервера: ${status}${detail ? ` — ${detail}` : ""}`;
};
