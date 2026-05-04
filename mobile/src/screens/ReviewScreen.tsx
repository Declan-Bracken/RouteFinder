import React, { useState, useCallback, useEffect } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  StyleSheet,
  ScrollView,
  useWindowDimensions,
} from "react-native";
import {
  getPendingImages,
  getPendingAreas,
  getPendingRoutes,
  reviewImage,
  reviewArea,
  reviewRoute,
  PendingImage,
  PendingArea,
  PendingRoute,
  RouteResult,
} from "../api/client";
import AreaRouteSearch from "../components/AreaRouteSearch";

type Step = "loading" | "review" | "correcting" | "done";
type Tab  = "images" | "areas" | "routes";

export default function ReviewScreen() {
  const { width } = useWindowDimensions();
  const [step, setStep]       = useState<Step>("loading");
  const [activeTab, setActiveTab] = useState<Tab>("images");

  const [queue, setQueue]     = useState<PendingImage[]>([]);
  const [index, setIndex]     = useState(0);
  const [submitting, setSubmitting] = useState(false);

  const [areas, setAreas]   = useState<PendingArea[]>([]);
  const [routes, setRoutes] = useState<PendingRoute[]>([]);

  const [correctionQuery, setCorrectionQuery]   = useState("");
  const [correctedRoute, setCorrectedRoute]     = useState<RouteResult | null>(null);

  useEffect(() => {
    Promise.all([getPendingImages(), getPendingAreas(), getPendingRoutes()])
      .then(([imgs, areasRes, routesRes]) => {
        setQueue(imgs.images);
        setAreas(areasRes.areas);
        setRoutes(routesRes.routes);
        const hasAny = imgs.images.length > 0 || areasRes.areas.length > 0 || routesRes.routes.length > 0;
        if (hasAny) {
          setActiveTab(imgs.images.length > 0 ? "images" : areasRes.areas.length > 0 ? "areas" : "routes");
          setStep("review");
        } else {
          setStep("done");
        }
      })
      .catch((e) => Alert.alert("Error", e.message));
  }, []);

  const current = queue[index] ?? null;

  const advance = useCallback(() => {
    setCorrectionQuery("");
    setCorrectedRoute(null);
    if (index + 1 >= queue.length) {
      setStep(areas.length > 0 || routes.length > 0 ? "review" : "done");
      if (areas.length > 0) setActiveTab("areas");
      else if (routes.length > 0) setActiveTab("routes");
    } else {
      setIndex((i) => i + 1);
      setStep("review");
    }
  }, [index, queue.length, areas.length, routes.length]);

  const handleImageAction = useCallback(
    async (action: "approve" | "reject", routeId?: number) => {
      if (!current) return;
      setSubmitting(true);
      try {
        await reviewImage(current.id, action, routeId);
        advance();
      } catch (e: any) {
        Alert.alert("Error", e.message);
      } finally {
        setSubmitting(false);
      }
    },
    [current, advance],
  );

  const handleAreaAction = useCallback(async (area: PendingArea, action: "approve" | "reject") => {
    try {
      await reviewArea(area.id, action);
      setAreas((prev) => prev.filter((a) => a.id !== area.id));
    } catch (e: any) {
      Alert.alert("Error", e.message);
    }
  }, []);

  const handleRouteAction = useCallback(async (route: PendingRoute, action: "approve" | "reject") => {
    try {
      await reviewRoute(route.id, action);
      setRoutes((prev) => prev.filter((r) => r.id !== route.id));
    } catch (e: any) {
      Alert.alert("Error", e.message);
    }
  }, []);

  // ── Loading ──────────────────────────────────────────────────────────────────
  if (step === "loading") {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#2563eb" />
        <Text style={styles.statusText}>Loading queue…</Text>
      </View>
    );
  }

  // ── Done ─────────────────────────────────────────────────────────────────────
  if (step === "done") {
    return (
      <View style={styles.centered}>
        <Text style={styles.doneIcon}>✓</Text>
        <Text style={styles.doneTitle}>All caught up!</Text>
        <Text style={styles.doneSubtitle}>No pending items to review.</Text>
      </View>
    );
  }

  // ── Route correction ─────────────────────────────────────────────────────────
  if (step === "correcting") {
    return (
      <View style={styles.fill}>
        <View style={styles.container}>
          <Text style={styles.heading}>Select correct route</Text>
          <Text style={styles.subtitle}>
            Currently tagged as: <Text style={styles.bold}>{current?.route_name}</Text>
          </Text>
          <AreaRouteSearch
            value={correctionQuery}
            onChangeText={setCorrectionQuery}
            placeholder="Search for the correct route…"
            showRoutes
            onSelectArea={() => {}}
            onSelectRoute={(route) => {
              setCorrectedRoute(route);
              setCorrectionQuery(route.name);
            }}
          />
          {correctedRoute && (
            <View style={styles.correctedCard}>
              <Text style={styles.correctedName}>{correctedRoute.name}</Text>
              <Text style={styles.correctedMeta}>
                {correctedRoute.grade ? `${correctedRoute.grade} · ` : ""}{correctedRoute.area}
              </Text>
            </View>
          )}
        </View>
        <View style={styles.footer}>
          <TouchableOpacity
            style={[styles.approveButton, !correctedRoute && styles.buttonDisabled]}
            disabled={!correctedRoute || submitting}
            onPress={() => handleImageAction("approve", correctedRoute!.id)}
          >
            {submitting
              ? <ActivityIndicator color="#fff" />
              : <Text style={styles.approveText}>Approve with correction</Text>}
          </TouchableOpacity>
          <TouchableOpacity style={styles.cancelLink} onPress={() => setStep("review")}>
            <Text style={styles.cancelLinkText}>← Back</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // ── Tab bar ──────────────────────────────────────────────────────────────────
  const tabCounts: Record<Tab, number> = {
    images: queue.length - index,
    areas:  areas.length,
    routes: routes.length,
  };

  const TabBar = () => (
    <View style={styles.tabBar}>
      {(["images", "areas", "routes"] as Tab[]).map((tab) => (
        <TouchableOpacity
          key={tab}
          style={[styles.tab, activeTab === tab && styles.tabActive]}
          onPress={() => setActiveTab(tab)}
        >
          <Text style={[styles.tabText, activeTab === tab && styles.tabTextActive]}>
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
            {tabCounts[tab] > 0 ? ` (${tabCounts[tab]})` : ""}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  // ── Areas tab ─────────────────────────────────────────────────────────────────
  if (activeTab === "areas") {
    return (
      <View style={styles.fill}>
        <TabBar />
        {areas.length === 0 ? (
          <View style={styles.centered}>
            <Text style={styles.emptyText}>No pending areas.</Text>
          </View>
        ) : (
          <ScrollView contentContainerStyle={styles.listContainer}>
            {areas.map((area) => (
              <View key={area.id} style={styles.catalogCard}>
                <View style={styles.catalogInfo}>
                  <Text style={styles.catalogName}>{area.name}</Text>
                  {area.parent_name && (
                    <Text style={styles.catalogMeta}>Under: {area.parent_name}</Text>
                  )}
                  <Text style={styles.catalogSubmitted}>
                    {area.submitted_by || "anonymous"}
                    {area.created_at ? ` · ${new Date(area.created_at).toLocaleDateString()}` : ""}
                  </Text>
                </View>
                <View style={styles.catalogActions}>
                  <TouchableOpacity
                    style={styles.rejectSmall}
                    onPress={() => handleAreaAction(area, "reject")}
                  >
                    <Text style={styles.actionTextSmall}>✕</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={styles.approveSmall}
                    onPress={() => handleAreaAction(area, "approve")}
                  >
                    <Text style={styles.actionTextSmall}>✓</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ))}
          </ScrollView>
        )}
      </View>
    );
  }

  // ── Routes tab ────────────────────────────────────────────────────────────────
  if (activeTab === "routes") {
    return (
      <View style={styles.fill}>
        <TabBar />
        {routes.length === 0 ? (
          <View style={styles.centered}>
            <Text style={styles.emptyText}>No pending routes.</Text>
          </View>
        ) : (
          <ScrollView contentContainerStyle={styles.listContainer}>
            {routes.map((route) => (
              <View key={route.id} style={styles.catalogCard}>
                <View style={styles.catalogInfo}>
                  <Text style={styles.catalogName}>{route.name}</Text>
                  <Text style={styles.catalogMeta}>
                    {route.grade ? `${route.grade} · ` : ""}{route.area}
                  </Text>
                  <Text style={styles.catalogSubmitted}>
                    {route.submitted_by || "anonymous"}
                    {route.created_at ? ` · ${new Date(route.created_at).toLocaleDateString()}` : ""}
                  </Text>
                </View>
                <View style={styles.catalogActions}>
                  <TouchableOpacity
                    style={styles.rejectSmall}
                    onPress={() => handleRouteAction(route, "reject")}
                  >
                    <Text style={styles.actionTextSmall}>✕</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={styles.approveSmall}
                    onPress={() => handleRouteAction(route, "approve")}
                  >
                    <Text style={styles.actionTextSmall}>✓</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ))}
          </ScrollView>
        )}
      </View>
    );
  }

  // ── Images tab ────────────────────────────────────────────────────────────────
  if (!current) {
    return (
      <View style={styles.fill}>
        <TabBar />
        <View style={styles.centered}>
          <Text style={styles.emptyText}>No pending images.</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.fill}>
      <TabBar />
      <Text style={styles.counter}>{tabCounts.images} remaining</Text>

      <Image
        source={{ uri: current.url }}
        style={styles.image}
        resizeMode="contain"
      />

      <View style={styles.infoCard}>
        <Text style={styles.routeName}>{current.route_name}</Text>
        <Text style={styles.routeMeta}>
          {current.grade ? `${current.grade} · ` : ""}{current.area}
        </Text>
        <Text style={styles.submittedBy}>
          Submitted by {current.submitted_by || "anonymous"}
          {current.created_at ? ` · ${new Date(current.created_at).toLocaleDateString()}` : ""}
        </Text>
      </View>

      <TouchableOpacity
        style={styles.correctLink}
        onPress={() => setStep("correcting")}
      >
        <Text style={styles.correctLinkText}>Wrong route? Correct before approving</Text>
      </TouchableOpacity>

      <View style={styles.footer}>
        <TouchableOpacity
          style={[styles.rejectButton, submitting && styles.buttonDisabled]}
          disabled={submitting}
          onPress={() => handleImageAction("reject")}
        >
          {submitting
            ? <ActivityIndicator color="#fff" />
            : <Text style={styles.rejectText}>✕  Reject</Text>}
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.approveButton, submitting && styles.buttonDisabled]}
          disabled={submitting}
          onPress={() => handleImageAction("approve")}
        >
          {submitting
            ? <ActivityIndicator color="#fff" />
            : <Text style={styles.approveText}>✓  Approve</Text>}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: "#fff" },
  centered: {
    flex: 1, justifyContent: "center", alignItems: "center",
    gap: 16, backgroundColor: "#fff",
  },
  container: { padding: 20 },

  // Tab bar
  tabBar: {
    flexDirection: "row",
    borderBottomWidth: 1,
    borderBottomColor: "#e5e7eb",
    backgroundColor: "#fff",
  },
  tab: {
    flex: 1, paddingVertical: 12, alignItems: "center",
  },
  tabActive: {
    borderBottomWidth: 2, borderBottomColor: "#2563eb",
  },
  tabText: { fontSize: 14, color: "#9ca3af", fontWeight: "600" },
  tabTextActive: { color: "#2563eb" },

  // Image review
  counter: { fontSize: 13, color: "#9ca3af", marginTop: 12, marginBottom: 8, textAlign: "center" },
  image: { flex: 1, width: "100%", borderRadius: 12, marginBottom: 12, backgroundColor: "#f3f4f6" },
  infoCard: {
    padding: 14, marginHorizontal: 16, borderRadius: 10,
    backgroundColor: "#f9fafb", borderWidth: 1, borderColor: "#e5e7eb",
    marginBottom: 8,
  },
  routeName: { fontSize: 17, fontWeight: "700" },
  routeMeta: { fontSize: 14, color: "#6b7280", marginTop: 2 },
  submittedBy: { fontSize: 12, color: "#9ca3af", marginTop: 6 },
  correctLink: { alignItems: "center", paddingVertical: 8 },
  correctLinkText: { color: "#2563eb", fontSize: 14 },

  footer: {
    flexDirection: "row", gap: 12,
    paddingVertical: 16, paddingBottom: 32,
    paddingHorizontal: 16,
    borderTopWidth: 1, borderTopColor: "#f3f4f6",
    backgroundColor: "#fff",
  },
  rejectButton: {
    flex: 1, padding: 16, borderRadius: 10,
    backgroundColor: "#dc2626", alignItems: "center",
  },
  rejectText: { color: "#fff", fontWeight: "700", fontSize: 16 },
  approveButton: {
    flex: 1, padding: 16, borderRadius: 10,
    backgroundColor: "#16a34a", alignItems: "center",
  },
  approveText: { color: "#fff", fontWeight: "700", fontSize: 16 },
  buttonDisabled: { opacity: 0.5 },
  cancelLink: { alignItems: "center", paddingVertical: 12 },
  cancelLinkText: { color: "#2563eb", fontSize: 14 },

  // Catalog (areas + routes) lists
  listContainer: { padding: 16, gap: 10 },
  catalogCard: {
    flexDirection: "row", alignItems: "center",
    padding: 14, borderRadius: 10,
    borderWidth: 1, borderColor: "#e5e7eb",
    backgroundColor: "#f9fafb",
    gap: 12,
  },
  catalogInfo: { flex: 1 },
  catalogName: { fontSize: 15, fontWeight: "600" },
  catalogMeta: { fontSize: 13, color: "#6b7280", marginTop: 2 },
  catalogSubmitted: { fontSize: 11, color: "#9ca3af", marginTop: 4 },
  catalogActions: { flexDirection: "row", gap: 8 },
  rejectSmall: {
    width: 36, height: 36, borderRadius: 8,
    backgroundColor: "#fee2e2", alignItems: "center", justifyContent: "center",
  },
  approveSmall: {
    width: 36, height: 36, borderRadius: 8,
    backgroundColor: "#dcfce7", alignItems: "center", justifyContent: "center",
  },
  actionTextSmall: { fontSize: 16, fontWeight: "700" },
  emptyText: { fontSize: 15, color: "#9ca3af" },

  // Correction step
  heading: { fontSize: 20, fontWeight: "700", marginBottom: 4 },
  subtitle: { fontSize: 14, color: "#6b7280", marginBottom: 16 },
  bold: { fontWeight: "600", color: "#111827" },
  correctedCard: {
    marginTop: 12, padding: 12, borderRadius: 10,
    backgroundColor: "#f0fdf4", borderWidth: 1, borderColor: "#16a34a",
  },
  correctedName: { fontSize: 15, fontWeight: "600" },
  correctedMeta: { fontSize: 13, color: "#6b7280", marginTop: 2 },

  statusText: { fontSize: 16, color: "#6b7280" },
  doneIcon: { fontSize: 56, color: "#16a34a" },
  doneTitle: { fontSize: 24, fontWeight: "700" },
  doneSubtitle: { fontSize: 16, color: "#6b7280" },
});
