import { Platform } from "react-native";

const BASE_URL = "https://routefinder-production.up.railway.app";

async function appendImageToForm(form: FormData, imageUri: string, fieldName = "file") {
  if (Platform.OS === "web") {
    const res = await fetch(imageUri);
    const blob = await res.blob();
    form.append(fieldName, blob, "photo.jpg");
  } else {
    form.append(fieldName, { uri: imageUri, name: "photo.jpg", type: "image/jpeg" } as any);
  }
}

export interface AreaResult {
  id: number;
  name: string;
  full_path: string;
  route_count: number;
}

export interface RouteResult {
  id: number;
  name: string;
  grade: string;
  area: string;
}

export interface SearchResults {
  areas: AreaResult[];
  routes: RouteResult[];
}

export interface RouteDetail {
  id: number;
  name: string;
  grade: string;
  url: string;
  area: string;
}

export interface ImageSearchResult {
  route_id: number;
  name: string;
  grade: string;
  url: string;
  area: string;
  similarity: number;
}

export async function unifiedSearch(q: string): Promise<SearchResults> {
  const res = await fetch(`${BASE_URL}/search?q=${encodeURIComponent(q)}`);
  if (!res.ok) throw new Error("Search failed");
  return res.json();
}

export async function searchByImage(
  imageUri: string,
  areaId: number | null = null,
): Promise<ImageSearchResult[]> {
  const form = new FormData();
  await appendImageToForm(form, imageUri);
  const url = areaId !== null
    ? `${BASE_URL}/search?area_id=${areaId}`
    : `${BASE_URL}/search`;
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Image search failed: ${text}`);
  }
  const data = await res.json();
  return data.results;
}

export async function getRoutes(areaId: number): Promise<RouteDetail[]> {
  const res = await fetch(`${BASE_URL}/areas/${areaId}/routes`);
  if (!res.ok) throw new Error("Failed to fetch routes");
  return res.json();
}

export async function submitImage(
  imageUri: string,
  routeId: number,
  source: "user" | "admin" = "user"
): Promise<{ id: string; route_id: number; status: string }> {
  const form = new FormData();
  await appendImageToForm(form, imageUri);
  form.append("route_id", String(routeId));
  form.append("source", source);

  const res = await fetch(`${BASE_URL}/images/submit`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Submit failed: ${text}`);
  }
  return res.json();
}
