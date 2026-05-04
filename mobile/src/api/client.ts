import { Platform } from "react-native";
import { ADMIN_API_KEY } from "../config";

const BASE_URL = "https://routefinder-production.up.railway.app";

const adminHeaders = () => ({ "X-Admin-Key": ADMIN_API_KEY });

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

export interface PendingImage {
  id: string;
  url: string;
  submitted_by: string;
  created_at: string | null;
  route_id: number;
  route_name: string;
  grade: string;
  area: string;
}

export async function getPendingImages(): Promise<{ count: number; images: PendingImage[] }> {
  const res = await fetch(`${BASE_URL}/admin/review/pending`, { headers: adminHeaders() });
  if (!res.ok) throw new Error("Failed to fetch pending images");
  return res.json();
}

export async function reviewImage(
  imageId: string,
  action: "approve" | "reject",
  correctRouteId?: number,
): Promise<void> {
  const res = await fetch(`${BASE_URL}/admin/review/${imageId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...adminHeaders() },
    body: JSON.stringify({ action, correct_route_id: correctRouteId ?? null }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Review failed: ${text}`);
  }
}

// ── Suggestions ──────────────────────────────────────────────────────────────

export async function suggestArea(
  name: string,
  parentId?: number,
): Promise<{ id: number }> {
  const res = await fetch(`${BASE_URL}/suggest/area`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, parent_id: parentId ?? null }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Suggest area failed: ${text}`);
  }
  return res.json();
}

export async function suggestRoute(
  name: string,
  grade: string,
  areaId: number,
): Promise<{ id: number }> {
  const res = await fetch(`${BASE_URL}/suggest/route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, grade, area_id: areaId }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Suggest route failed: ${text}`);
  }
  return res.json();
}

// ── Admin: pending areas / routes ─────────────────────────────────────────────

export interface PendingArea {
  id: number;
  name: string;
  parent_id: number | null;
  parent_name: string | null;
  submitted_by: string | null;
  created_at: string | null;
}

export interface PendingRoute {
  id: number;
  name: string;
  grade: string;
  area_id: number;
  area: string;
  submitted_by: string | null;
  created_at: string | null;
}

export async function getPendingAreas(): Promise<{ count: number; areas: PendingArea[] }> {
  const res = await fetch(`${BASE_URL}/admin/review/pending/areas`, { headers: adminHeaders() });
  if (!res.ok) throw new Error("Failed to fetch pending areas");
  return res.json();
}

export async function getPendingRoutes(): Promise<{ count: number; routes: PendingRoute[] }> {
  const res = await fetch(`${BASE_URL}/admin/review/pending/routes`, { headers: adminHeaders() });
  if (!res.ok) throw new Error("Failed to fetch pending routes");
  return res.json();
}

export async function reviewArea(areaId: number, action: "approve" | "reject"): Promise<void> {
  const res = await fetch(`${BASE_URL}/admin/review/areas/${areaId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...adminHeaders() },
    body: JSON.stringify({ action }),
  });
  if (!res.ok) throw new Error("Area review failed");
}

export async function reviewRoute(routeId: number, action: "approve" | "reject"): Promise<void> {
  const res = await fetch(`${BASE_URL}/admin/review/routes/${routeId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...adminHeaders() },
    body: JSON.stringify({ action }),
  });
  if (!res.ok) throw new Error("Route review failed");
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
