import React, { useState, useCallback } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  StyleSheet,
  ScrollView,
  Platform,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { searchByImage, submitImage, AreaResult, ImageSearchResult } from "../api/client";
import AreaRouteSearch from "../components/AreaRouteSearch";

type Step = "area" | "photo" | "searching" | "results" | "confirm" | "submitting" | "done";

const ROUTE_COUNT_WARN  = 500;
const ROUTE_COUNT_LIMIT = 1000;

function SimilarityBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.9 ? "#16a34a" : value >= 0.75 ? "#d97706" : "#dc2626";
  return (
    <View style={[badge.wrap, { backgroundColor: color + "20", borderColor: color }]}>
      <Text style={[badge.text, { color }]}>{pct}%</Text>
    </View>
  );
}
const badge = StyleSheet.create({
  wrap: { borderWidth: 1, borderRadius: 6, paddingHorizontal: 8, paddingVertical: 3 },
  text: { fontSize: 13, fontWeight: "700" },
});

export default function IdentifyScreen() {
  const [step, setStep] = useState<Step>("area");

  // Area selection
  const [areaQuery, setAreaQuery]       = useState("");
  const [selectedArea, setSelectedArea] = useState<AreaResult | null>(null);

  // Image search
  const [imageUri, setImageUri]   = useState<string | null>(null);
  const [results, setResults]     = useState<ImageSearchResult[]>([]);
  const [selected, setSelected]   = useState<ImageSearchResult | null>(null);

  const reset = useCallback(() => {
    setStep("area");
    setAreaQuery("");
    setAreaResults([]);
    setSelectedArea(null);
    setImageUri(null);
    setResults([]);
    setSelected(null);
  }, []);

  const selectArea = useCallback((area: AreaResult) => {
    setSelectedArea(area);
    setAreaQuery(area.full_path);
  }, []);

  const clearArea = useCallback(() => {
    setSelectedArea(null);
    setAreaQuery("");
  }, []);

  // ── Photo & search ───────────────────────────────────────────────────────────
  const runSearch = useCallback(async (uri: string) => {
    setImageUri(uri);
    setStep("searching");
    try {
      const res = await searchByImage(uri, selectedArea?.id ?? null);
      setResults(res);
      setStep("results");
    } catch (e: any) {
      Alert.alert("Search failed", e.message);
      setStep("photo");
    }
  }, [selectedArea]);

  const pickImage = useCallback(async () => {
    if (Platform.OS !== "web") {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== "granted") {
        Alert.alert("Permission needed", "Photo library access is required.");
        return;
      }
    }
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ["images"], quality: 0.9 });
    if (!result.canceled && result.assets?.[0]) runSearch(result.assets[0].uri);
  }, [runSearch]);

  const takePhoto = useCallback(async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission needed", "Camera access is required.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ quality: 0.9 });
    if (!result.canceled && result.assets?.[0]) runSearch(result.assets[0].uri);
  }, [runSearch]);

  // ── Submit confirmed match ───────────────────────────────────────────────────
  const handleConfirm = useCallback(async () => {
    if (!imageUri || !selected) return;
    setStep("submitting");
    try {
      await submitImage(imageUri, selected.route_id);
      setStep("done");
    } catch (e: any) {
      Alert.alert("Submit failed", e.message);
      setStep("confirm");
    }
  }, [imageUri, selected]);

  // ── Searching ────────────────────────────────────────────────────────────────
  if (step === "searching") {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#2563eb" />
        <Text style={styles.statusText}>Identifying route…</Text>
        {selectedArea && (
          <Text style={styles.statusSub}>Searching within {selectedArea.name}</Text>
        )}
      </View>
    );
  }

  // ── Submitting ───────────────────────────────────────────────────────────────
  if (step === "submitting") {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#2563eb" />
        <Text style={styles.statusText}>Uploading…</Text>
      </View>
    );
  }

  // ── Done ─────────────────────────────────────────────────────────────────────
  if (step === "done") {
    return (
      <View style={styles.centered}>
        <Text style={styles.doneIcon}>✓</Text>
        <Text style={styles.doneTitle}>Photo submitted!</Text>
        <Text style={styles.doneSubtitle}>
          {selected?.name}{selected?.grade ? ` · ${selected.grade}` : ""}
        </Text>
        <TouchableOpacity style={styles.primaryButton} onPress={reset}>
          <Text style={styles.primaryButtonText}>Identify another</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── Confirm ──────────────────────────────────────────────────────────────────
  if (step === "confirm" && selected && imageUri) {
    return (
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.heading}>Confirm route</Text>
        <View style={styles.routeCard}>
          <View style={styles.routeCardRow}>
            <View style={{ flex: 1 }}>
              <Text style={styles.routeName}>{selected.name}</Text>
              <Text style={styles.routeMeta}>
                {selected.grade ? `${selected.grade} · ` : ""}{selected.area}
              </Text>
            </View>
            <SimilarityBadge value={selected.similarity} />
          </View>
        </View>
        <Image source={{ uri: imageUri }} style={styles.preview} resizeMode="cover" />
        <TouchableOpacity style={styles.primaryButton} onPress={handleConfirm}>
          <Text style={styles.primaryButtonText}>Yes, submit this photo</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.backLink} onPress={() => setStep("results")}>
          <Text style={styles.backLinkText}>← Back to results</Text>
        </TouchableOpacity>
      </ScrollView>
    );
  }

  // ── Results ──────────────────────────────────────────────────────────────────
  if (step === "results") {
    return (
      <View style={styles.fill}>
        <View style={styles.resultsHeader}>
          <View>
            <Text style={styles.heading}>Top matches</Text>
            {selectedArea && (
              <Text style={styles.resultsScope}>within {selectedArea.name}</Text>
            )}
          </View>
          {imageUri && (
            <Image source={{ uri: imageUri }} style={styles.thumbnail} resizeMode="cover" />
          )}
        </View>
        <ScrollView contentContainerStyle={styles.resultsList}>
          {results.map((r, i) => (
            <TouchableOpacity
              key={r.route_id + "-" + i}
              style={styles.resultCard}
              onPress={() => { setSelected(r); setStep("confirm"); }}
            >
              <View style={styles.resultCardLeft}>
                <Text style={styles.resultRank}>#{i + 1}</Text>
                <View style={{ flex: 1 }}>
                  <Text style={styles.routeName}>{r.name}</Text>
                  <Text style={styles.routeMeta}>
                    {r.grade ? `${r.grade} · ` : ""}{r.area}
                  </Text>
                </View>
              </View>
              <SimilarityBadge value={r.similarity} />
            </TouchableOpacity>
          ))}
          <TouchableOpacity style={styles.retryButton} onPress={reset}>
            <Text style={styles.retryButtonText}>Try a different photo</Text>
          </TouchableOpacity>
        </ScrollView>
      </View>
    );
  }

  // ── Photo ────────────────────────────────────────────────────────────────────
  if (step === "photo") {
    return (
      <View style={styles.centered}>
        {selectedArea ? (
          <View style={styles.areaChip}>
            <Text style={styles.areaChipText} numberOfLines={1}>{selectedArea.full_path}</Text>
            <TouchableOpacity onPress={clearArea} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
              <Text style={styles.areaChipX}>✕</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <Text style={styles.subtitle}>Searching all areas</Text>
        )}
        <TouchableOpacity style={styles.primaryButton} onPress={takePhoto}>
          <Text style={styles.primaryButtonText}>Take a photo</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.secondaryButton} onPress={pickImage}>
          <Text style={styles.secondaryButtonText}>Choose from library</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.backLink} onPress={() => setStep("area")}>
          <Text style={styles.backLinkText}>← Change area</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── Area selection ───────────────────────────────────────────────────────────
  const showWarning = selectedArea && selectedArea.route_count > ROUTE_COUNT_WARN;

  return (
    <View style={styles.fill}>
      <View style={styles.container}>
        <Text style={styles.heading}>Where are you climbing?</Text>
        <Text style={styles.subtitle}>Narrow the search to your crag for better results</Text>
        <AreaRouteSearch
          value={areaQuery}
          onChangeText={setAreaQuery}
          placeholder="Search area or crag…"
          showRoutes={false}
          routeCountWarn={ROUTE_COUNT_WARN}
          routeCountLimit={ROUTE_COUNT_LIMIT}
          onSelectArea={selectArea}
        />
        {showWarning && (
          <View style={styles.warningBox}>
            <Text style={styles.warningText}>
              {selectedArea!.name} has {selectedArea!.route_count} routes — results may be less precise.
            </Text>
          </View>
        )}
      </View>

      <View style={styles.areaFooter}>
        <TouchableOpacity style={styles.primaryButton} onPress={() => setStep("photo")}>
          <Text style={styles.primaryButtonText}>
            {selectedArea ? `Search in ${selectedArea.name}` : "Search everywhere"}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: "#fff" },
  centered: {
    flex: 1, justifyContent: "center", alignItems: "center",
    gap: 16, paddingHorizontal: 24, backgroundColor: "#fff",
  },
  container: { padding: 20, paddingBottom: 8 },
  heading: { fontSize: 22, fontWeight: "700", marginBottom: 4 },
  subtitle: { fontSize: 15, color: "#6b7280", marginBottom: 16 },
  statusText: { fontSize: 16, color: "#6b7280" },
  statusSub: { fontSize: 13, color: "#9ca3af" },

  warningBox: {
    marginTop: 10, padding: 10, borderRadius: 8,
    backgroundColor: "#fef9c3", borderWidth: 1, borderColor: "#fde047",
  },
  warningText: { fontSize: 13, color: "#854d0e" },
  areaFooter: { padding: 20, paddingTop: 8 },

  // Area chip (shown on photo step)
  areaChip: {
    flexDirection: "row", alignItems: "center", gap: 8,
    backgroundColor: "#eff6ff", borderWidth: 1, borderColor: "#2563eb",
    borderRadius: 20, paddingHorizontal: 14, paddingVertical: 8,
    maxWidth: "90%",
  },
  areaChipText: { fontSize: 14, fontWeight: "600", color: "#2563eb", flex: 1 },
  areaChipX: { fontSize: 14, color: "#2563eb" },

  // Results
  resultsHeader: {
    flexDirection: "row", alignItems: "center", justifyContent: "space-between",
    paddingHorizontal: 20, paddingTop: 20, paddingBottom: 12,
    borderBottomWidth: 1, borderBottomColor: "#f3f4f6",
  },
  resultsScope: { fontSize: 13, color: "#6b7280", marginTop: 2 },
  thumbnail: { width: 56, height: 56, borderRadius: 8 },
  resultsList: { padding: 16 },
  resultCard: {
    flexDirection: "row", alignItems: "center", justifyContent: "space-between",
    padding: 14, borderRadius: 10, borderWidth: 1, borderColor: "#d1d5db",
    backgroundColor: "#fff", marginBottom: 8,
  },
  resultCardLeft: { flexDirection: "row", alignItems: "center", flex: 1, gap: 10, marginRight: 10 },
  resultRank: { fontSize: 13, fontWeight: "700", color: "#9ca3af", width: 24 },

  // Confirm / shared route card
  routeCard: {
    padding: 14, borderRadius: 10, backgroundColor: "#eff6ff",
    borderWidth: 1, borderColor: "#2563eb", marginBottom: 16,
  },
  routeCardRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  routeName: { fontSize: 15, fontWeight: "600" },
  routeMeta: { fontSize: 13, color: "#6b7280", marginTop: 2 },
  preview: { width: "100%", height: 260, borderRadius: 10, marginBottom: 16 },

  // Buttons
  primaryButton: {
    backgroundColor: "#2563eb", padding: 14, borderRadius: 10,
    alignItems: "center", width: "100%",
  },
  primaryButtonText: { color: "#fff", fontWeight: "700", fontSize: 16 },
  secondaryButton: {
    padding: 14, borderRadius: 10, borderWidth: 1, borderColor: "#2563eb",
    alignItems: "center", width: "100%",
  },
  secondaryButtonText: { color: "#2563eb", fontWeight: "600", fontSize: 15 },
  retryButton: { padding: 14, alignItems: "center", marginTop: 4 },
  retryButtonText: { color: "#6b7280", fontSize: 14 },
  backLink: { padding: 12, alignItems: "center" },
  backLinkText: { color: "#2563eb", fontSize: 14 },

  // Done
  doneIcon: { fontSize: 56, color: "#16a34a" },
  doneTitle: { fontSize: 24, fontWeight: "700" },
  doneSubtitle: { fontSize: 16, color: "#6b7280" },
});
