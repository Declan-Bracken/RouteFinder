const BASE_URL = "https://routefinder-production.up.railway.app";

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

export async function unifiedSearch(q: string): Promise<SearchResults> {
  const res = await fetch(`${BASE_URL}/search?q=${encodeURIComponent(q)}`);
  if (!res.ok) throw new Error("Search failed");
  return res.json();
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
): Promise<{ image_id: string }> {
  const form = new FormData();
  form.append("file", {
    uri: imageUri,
    name: "photo.jpg",
    type: "image/jpeg",
  } as any);
  form.append("route_id", String(routeId));
  form.append("source", source);

  const res = await fetch(`${BASE_URL}/submit`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Submit failed: ${text}`);
  }
  return res.json();
}
