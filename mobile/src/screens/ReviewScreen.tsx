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
import { getPendingImages, reviewImage, PendingImage, RouteResult } from "../api/client";
import AreaRouteSearch from "../components/AreaRouteSearch";

type Step = "loading" | "review" | "correcting" | "done";

export default function ReviewScreen() {
  const [step, setStep]       = useState<Step>("loading");
  const [queue, setQueue]     = useState<PendingImage[]>([]);
  const [index, setIndex]     = useState(0);
  const [submitting, setSubmitting] = useState(false);

  // Route correction state
  const [correctionQuery, setCorrectionQuery]   = useState("");
  const [correctedRoute, setCorrectedRoute]     = useState<RouteResult | null>(null);

  useEffect(() => {
    getPendingImages()
      .then(({ images }) => {
        setQueue(images);
        setStep(images.length > 0 ? "review" : "done");
      })
      .catch((e) => Alert.alert("Error", e.message));
  }, []);

  const current = queue[index] ?? null;

  const advance = useCallback(() => {
    setCorrectionQuery("");
    setCorrectedRoute(null);
    if (index + 1 >= queue.length) {
      setStep("done");
    } else {
      setIndex((i) => i + 1);
      setStep("review");
    }
  }, [index, queue.length]);

  const handleAction = useCallback(
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
  if (step === "done" || !current) {
    return (
      <View style={styles.centered}>
        <Text style={styles.doneIcon}>✓</Text>
        <Text style={styles.doneTitle}>All caught up!</Text>
        <Text style={styles.doneSubtitle}>No pending images to review.</Text>
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
            Currently tagged as: <Text style={styles.bold}>{current.route_name}</Text>
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
            onPress={() => handleAction("approve", correctedRoute!.id)}
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

  // ── Review card ──────────────────────────────────────────────────────────────
  const remaining = queue.length - index;
  const { width } = useWindowDimensions();
  const imageHeight = Math.round(width * 0.75); // 4:3 aspect, full width

  return (
    <View style={styles.fill}>
      <ScrollView contentContainerStyle={styles.cardContainer}>
        <Text style={styles.counter}>{remaining} remaining</Text>

        <Image
          source={{ uri: current.url }}
          style={[styles.image, { height: imageHeight }]}
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
      </ScrollView>

      <View style={styles.footer}>
        <TouchableOpacity
          style={[styles.rejectButton, submitting && styles.buttonDisabled]}
          disabled={submitting}
          onPress={() => handleAction("reject")}
        >
          {submitting
            ? <ActivityIndicator color="#fff" />
            : <Text style={styles.rejectText}>✕  Reject</Text>}
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.approveButton, submitting && styles.buttonDisabled]}
          disabled={submitting}
          onPress={() => handleAction("approve")}
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
  cardContainer: { padding: 20, paddingBottom: 120 },

  counter: { fontSize: 13, color: "#9ca3af", marginBottom: 12, textAlign: "center" },
  image: { width: "100%", borderRadius: 12, marginBottom: 16, backgroundColor: "#f3f4f6" },

  infoCard: {
    padding: 14, borderRadius: 10,
    backgroundColor: "#f9fafb", borderWidth: 1, borderColor: "#e5e7eb",
    marginBottom: 12,
  },
  routeName: { fontSize: 17, fontWeight: "700" },
  routeMeta: { fontSize: 14, color: "#6b7280", marginTop: 2 },
  submittedBy: { fontSize: 12, color: "#9ca3af", marginTop: 6 },

  correctLink: { alignItems: "center", paddingVertical: 8 },
  correctLinkText: { color: "#2563eb", fontSize: 14 },

  footer: {
    flexDirection: "row", gap: 12,
    padding: 16, paddingBottom: 32,
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
