import React, { useState, useCallback } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  TextInput,
  Image,
  ActivityIndicator,
  Alert,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  useWindowDimensions,
} from "react-native";

import * as ImagePicker from "expo-image-picker";
import {
  getRoutes,
  submitImage,
  suggestArea,
  suggestRoute,
  AreaResult,
  RouteResult,
  RouteDetail,
} from "../api/client";
import AreaRouteSearch from "../components/AreaRouteSearch";

type Step = "search" | "routes" | "photo" | "submitting" | "done" | "suggest_area" | "suggest_route";

export default function SubmitScreen() {
  const { height } = useWindowDimensions();
  const [step, setStep] = useState<Step>("search");

  const [query, setQuery] = useState("");

  const [selectedArea, setSelectedArea] = useState<AreaResult | null>(null);
  const [routes, setRoutes] = useState<RouteDetail[]>([]);
  const [routesLoading, setRoutesLoading] = useState(false);

  const [selectedRoute, setSelectedRoute] = useState<RouteDetail | RouteResult | null>(null);
  const [imageUris, setImageUris] = useState<string[]>([]);
  const [submitProgress, setSubmitProgress] = useState("");

  // Suggest area state
  const [suggestAreaName, setSuggestAreaName]       = useState("");
  const [suggestParentQuery, setSuggestParentQuery] = useState("");
  const [suggestParentId, setSuggestParentId]       = useState<number | null>(null);
  const [suggestAreaLoading, setSuggestAreaLoading] = useState(false);

  // Suggest route state
  const [suggestRouteName, setSuggestRouteName]   = useState("");
  const [suggestRouteGrade, setSuggestRouteGrade] = useState("");
  const [suggestRouteLoading, setSuggestRouteLoading] = useState(false);

  const selectArea = useCallback(async (area: AreaResult) => {
    setSelectedArea(area);
    setQuery(area.full_path);
    setRoutesLoading(true);
    setStep("routes");
    try {
      const r = await getRoutes(area.id);
      setRoutes(r);
    } catch {
      Alert.alert("Error", "Failed to load routes.");
    } finally {
      setRoutesLoading(false);
    }
  }, []);

  const selectRoute = useCallback((route: RouteDetail | RouteResult) => {
    setSelectedRoute(route);
    setStep("photo");
  }, []);

  // Go back to route list preserving area + routes state
  const backToRoutes = useCallback(() => {
    setSelectedRoute(null);
    setImageUris([]);
    setStep("routes");
  }, []);

  const resetAll = useCallback(() => {
    setStep("search");
    setQuery("");
    setSelectedArea(null);
    setRoutes([]);
    setSelectedRoute(null);
    setImageUris([]);
    setSubmitProgress("");
  }, []);

  const pickImage = useCallback(async () => {
    if (Platform.OS === "web") {
      const input = document.createElement("input");
      input.type = "file";
      input.multiple = true;
      input.accept = "image/*";
      input.onchange = (e) => {
        const files = (e.target as HTMLInputElement).files;
        if (files?.length) {
          const uris = Array.from(files).map((f) => URL.createObjectURL(f));
          setImageUris((prev) => [...prev, ...uris]);
        }
      };
      input.click();
      return;
    }
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission needed", "Photo library access is required.");
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      allowsMultipleSelection: true,
      quality: 0.9,
    });
    if (!result.canceled && result.assets?.length) {
      setImageUris((prev) => [...prev, ...result.assets.map((a) => a.uri)]);
    }
  }, []);

  const takePhoto = useCallback(async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission needed", "Camera access is required to take photos.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ quality: 0.9 });
    if (!result.canceled && result.assets?.[0]) {
      setImageUris((prev) => [...prev, result.assets[0].uri]);
    }
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!imageUris.length || !selectedRoute) return;
    setStep("submitting");
    let failed = 0;
    for (let i = 0; i < imageUris.length; i++) {
      setSubmitProgress(`${i + 1} / ${imageUris.length}`);
      try {
        await submitImage(imageUris[i], selectedRoute.id);
      } catch {
        failed++;
      }
    }
    if (failed > 0) {
      Alert.alert("Partial failure", `${failed} photo(s) failed to upload.`);
    }
    setStep("done");
  }, [imageUris, selectedRoute]);

  const handleSuggestArea = useCallback(async () => {
    if (!suggestAreaName.trim()) return;
    setSuggestAreaLoading(true);
    try {
      await suggestArea(suggestAreaName.trim(), suggestParentId ?? undefined);
      Alert.alert("Submitted!", "Your area suggestion will appear after review.");
      setSuggestAreaName(""); setSuggestParentQuery(""); setSuggestParentId(null);
      setStep("search");
    } catch (e: any) {
      Alert.alert("Error", e.message);
    } finally {
      setSuggestAreaLoading(false);
    }
  }, [suggestAreaName, suggestParentId]);

  const handleSuggestRoute = useCallback(async () => {
    if (!suggestRouteName.trim() || !suggestRouteGrade.trim() || !selectedArea) return;
    setSuggestRouteLoading(true);
    try {
      await suggestRoute(suggestRouteName.trim(), suggestRouteGrade.trim(), selectedArea.id);
      Alert.alert("Submitted!", "Your route suggestion will appear after review.");
      setSuggestRouteName(""); setSuggestRouteGrade("");
      setStep("routes");
    } catch (e: any) {
      Alert.alert("Error", e.message);
    } finally {
      setSuggestRouteLoading(false);
    }
  }, [suggestRouteName, suggestRouteGrade, selectedArea]);

  // ── Submitting ──────────────────────────────────────────────────────────────
  if (step === "submitting") {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#2563eb" />
        <Text style={styles.statusText}>
          Uploading{submitProgress ? ` ${submitProgress}` : "…"}
        </Text>
      </View>
    );
  }

  // ── Done ────────────────────────────────────────────────────────────────────
  if (step === "done") {
    return (
      <View style={styles.centered}>
        <Text style={styles.doneIcon}>✓</Text>
        <Text style={styles.doneTitle}>Submitted!</Text>
        <Text style={styles.doneSubtitle}>
          {selectedRoute?.name}
          {selectedRoute?.grade ? ` · ${selectedRoute.grade}` : ""}
        </Text>
        <TouchableOpacity style={styles.primaryButton} onPress={resetAll}>
          <Text style={styles.primaryButtonText}>Submit another</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── Routes step (FlatList for proper scrolling) ─────────────────────────────
  if (step === "routes") {
    return (
      <View style={styles.fill}>
        <View style={styles.routesHeader}>
          <Text style={styles.label}>Area</Text>
          <View style={styles.searchRow}>
            <Text style={styles.frozenQuery} numberOfLines={1}>{query}</Text>
            <TouchableOpacity onPress={resetAll} style={styles.clearButton}>
              <Text style={styles.clearButtonText}>✕</Text>
            </TouchableOpacity>
          </View>
          <Text style={[styles.label, { marginTop: 16 }]}>Route</Text>
        </View>

        {routesLoading ? (
          <ActivityIndicator style={{ marginTop: 24 }} />
        ) : (
          <ScrollView style={{ height: height - 130 }} contentContainerStyle={styles.routeListContent}>
            {routes.map((r) => (
              <TouchableOpacity key={r.id} style={styles.routeItem} onPress={() => selectRoute(r)}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.routeName}>{r.name}</Text>
                  <Text style={styles.routeArea}>{r.area}</Text>
                </View>
                <Text style={styles.routeGrade}>{r.grade}</Text>
              </TouchableOpacity>
            ))}
            <TouchableOpacity style={styles.suggestLink} onPress={() => setStep("suggest_route")}>
              <Text style={styles.suggestLinkText}>Route not listed? Suggest it →</Text>
            </TouchableOpacity>
          </ScrollView>
        )}
      </View>
    );
  }

  // ── Photo step ──────────────────────────────────────────────────────────────
  if (step === "photo" && selectedRoute) {
    return (
      <KeyboardAvoidingView
        style={styles.fill}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        <ScrollView style={{ flex: 1 }} contentContainerStyle={styles.container}>
          <Text style={styles.heading}>Submit a route photo</Text>

          <View style={styles.selectedRouteCard}>
            <Text style={styles.selectedRouteName}>{selectedRoute.name}</Text>
            <Text style={styles.selectedRouteMeta}>
              {selectedRoute.grade ? `${selectedRoute.grade} · ` : ""}
              {"area" in selectedRoute ? selectedRoute.area : ""}
            </Text>
          </View>
          <TouchableOpacity onPress={backToRoutes} style={styles.changeLink}>
            <Text style={styles.changeLinkText}>← Change route</Text>
          </TouchableOpacity>

          <View style={styles.tipsCard}>
            <Text style={styles.tipsTitle}>Photo tips</Text>
            <View style={styles.tipsColumns}>
              <View style={styles.tipsCol}>
                <Text style={styles.tipsGood}>✓  Do</Text>
                <Text style={styles.tipItem}>Shoot at a variety of angles — straight on, from below, or stepped back</Text>
                <Text style={styles.tipItem}>Show the key features that distinguish this route</Text>
              </View>
              <View style={styles.tipsDivider} />
              <View style={styles.tipsCol}>
                <Text style={styles.tipsBad}>✕  Don't</Text>
                <Text style={styles.tipItem}>Include multiple routes in the frame</Text>
                <Text style={styles.tipItem}>Shoot at an obstructive angle</Text>
                <Text style={styles.tipItem}>Get too close — the full line should be visible</Text>
              </View>
            </View>
          </View>

          <Text style={styles.label}>Photos</Text>
          <View style={styles.photoButtons}>
            <TouchableOpacity style={styles.secondaryButton} onPress={takePhoto}>
              <Text style={styles.secondaryButtonText}>Take photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.secondaryButton} onPress={pickImage}>
              <Text style={styles.secondaryButtonText}>
                {imageUris.length > 0 ? "Add more photos" : "Choose from library"}
              </Text>
            </TouchableOpacity>
          </View>
          {imageUris.length > 0 && (
            <>
              <View style={styles.thumbnailGrid}>
                {imageUris.map((uri, i) => (
                  <View key={uri + i} style={styles.thumbnailWrapper}>
                    <Image source={{ uri }} style={styles.thumbnail} resizeMode="cover" />
                    <TouchableOpacity
                      style={styles.thumbnailRemove}
                      onPress={() => setImageUris((prev) => prev.filter((_, j) => j !== i))}
                    >
                      <Text style={styles.thumbnailRemoveText}>✕</Text>
                    </TouchableOpacity>
                  </View>
                ))}
              </View>
              <TouchableOpacity style={styles.primaryButton} onPress={handleSubmit}>
                <Text style={styles.primaryButtonText}>
                  Submit {imageUris.length} photo{imageUris.length > 1 ? "s" : ""}
                </Text>
              </TouchableOpacity>
            </>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    );
  }

  // ── Suggest area ─────────────────────────────────────────────────────────────
  if (step === "suggest_area") {
    return (
      <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === "ios" ? "padding" : undefined}>
        <ScrollView style={{ flex: 1 }} contentContainerStyle={styles.container}>
          <Text style={styles.heading}>Suggest a new area</Text>
          <Text style={styles.label}>Area name</Text>
          <TextInput
            style={styles.textInput}
            placeholder="e.g. Limestone Wall"
            value={suggestAreaName}
            onChangeText={setSuggestAreaName}
            autoFocus
          />
          <Text style={[styles.label, { marginTop: 16 }]}>Parent area (optional)</Text>
          <AreaRouteSearch
            value={suggestParentQuery}
            onChangeText={setSuggestParentQuery}
            placeholder="Search for a parent area…"
            showRoutes={false}
            onSelectArea={(area) => { setSuggestParentId(area.id); setSuggestParentQuery(area.name); }}
            onSelectRoute={() => {}}
          />
          {suggestParentId && (
            <TouchableOpacity onPress={() => { setSuggestParentId(null); setSuggestParentQuery(""); }}>
              <Text style={styles.changeLinkText}>✕ Clear parent</Text>
            </TouchableOpacity>
          )}
          <TouchableOpacity
            style={[styles.primaryButton, { marginTop: 24 }, !suggestAreaName.trim() && styles.buttonDisabled]}
            disabled={!suggestAreaName.trim() || suggestAreaLoading}
            onPress={handleSuggestArea}
          >
            {suggestAreaLoading
              ? <ActivityIndicator color="#fff" />
              : <Text style={styles.primaryButtonText}>Submit for review</Text>}
          </TouchableOpacity>
          <TouchableOpacity style={styles.changeLink} onPress={() => setStep("search")}>
            <Text style={styles.changeLinkText}>← Back to search</Text>
          </TouchableOpacity>
        </ScrollView>
      </KeyboardAvoidingView>
    );
  }

  // ── Suggest route ─────────────────────────────────────────────────────────────
  if (step === "suggest_route") {
    return (
      <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === "ios" ? "padding" : undefined}>
        <ScrollView style={{ flex: 1 }} contentContainerStyle={styles.container}>
          <Text style={styles.heading}>Suggest a new route</Text>
          {selectedArea && (
            <View style={styles.selectedRouteCard}>
              <Text style={styles.selectedRouteName}>{selectedArea.name}</Text>
            </View>
          )}
          <Text style={[styles.label, { marginTop: 16 }]}>Route name</Text>
          <TextInput
            style={styles.textInput}
            placeholder="e.g. Crack Attack"
            value={suggestRouteName}
            onChangeText={setSuggestRouteName}
            autoFocus
          />
          <Text style={[styles.label, { marginTop: 16 }]}>Grade</Text>
          <TextInput
            style={styles.textInput}
            placeholder="e.g. 5.10a"
            value={suggestRouteGrade}
            onChangeText={setSuggestRouteGrade}
            autoCapitalize="none"
          />
          <TouchableOpacity
            style={[styles.primaryButton, { marginTop: 24 }, (!suggestRouteName.trim() || !suggestRouteGrade.trim()) && styles.buttonDisabled]}
            disabled={!suggestRouteName.trim() || !suggestRouteGrade.trim() || suggestRouteLoading}
            onPress={handleSuggestRoute}
          >
            {suggestRouteLoading
              ? <ActivityIndicator color="#fff" />
              : <Text style={styles.primaryButtonText}>Submit for review</Text>}
          </TouchableOpacity>
          <TouchableOpacity style={styles.changeLink} onPress={() => setStep("routes")}>
            <Text style={styles.changeLinkText}>← Back to routes</Text>
          </TouchableOpacity>
        </ScrollView>
      </KeyboardAvoidingView>
    );
  }

  // ── Search step ─────────────────────────────────────────────────────────────
  return (
    <KeyboardAvoidingView
      style={styles.fill}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <View style={styles.container}>
        <Text style={styles.heading}>Submit a route photo</Text>
        <Text style={styles.label}>Search area or route</Text>
        <AreaRouteSearch
          value={query}
          onChangeText={setQuery}
          placeholder="e.g. Mount Nemo, Lop Sided…"
          showRoutes
          onSelectArea={selectArea}
          onSelectRoute={selectRoute}
        />
        <TouchableOpacity style={styles.suggestLink} onPress={() => setStep("suggest_area")}>
          <Text style={styles.suggestLinkText}>Area not listed? Suggest it →</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  fill: {
    flex: 1,
    backgroundColor: "#fff",
  },
  container: {
    padding: 20,
    paddingBottom: 48,
  },
  centered: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: 16,
    backgroundColor: "#fff",
  },
  heading: {
    fontSize: 22,
    fontWeight: "700",
    marginBottom: 24,
    marginTop: 8,
  },
  label: {
    fontSize: 13,
    fontWeight: "600",
    color: "#6b7280",
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginBottom: 6,
  },
  searchRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  frozenQuery: {
    flex: 1,
    fontSize: 16,
    fontWeight: "500",
    paddingVertical: 4,
  },
  clearButton: {
    padding: 8,
  },
  clearButtonText: {
    fontSize: 16,
    color: "#9ca3af",
  },
  // Routes step
  routesHeader: {
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 8,
    backgroundColor: "#fff",
    borderBottomWidth: 1,
    borderBottomColor: "#f3f4f6",
  },
  routeListContent: {
    padding: 16,
    gap: 6,
  },
  routeItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#d1d5db",
    backgroundColor: "#fff",
    marginBottom: 6,
  },
  routeName: {
    fontSize: 15,
    fontWeight: "500",
  },
  routeArea: {
    fontSize: 13,
    color: "#6b7280",
    marginTop: 2,
  },
  routeGrade: {
    fontSize: 14,
    fontWeight: "600",
    color: "#6b7280",
    marginLeft: 8,
  },
  // Photo step
  selectedRouteCard: {
    padding: 14,
    borderRadius: 10,
    backgroundColor: "#eff6ff",
    borderWidth: 1,
    borderColor: "#2563eb",
    marginBottom: 4,
  },
  selectedRouteName: {
    fontSize: 16,
    fontWeight: "600",
  },
  selectedRouteMeta: {
    fontSize: 13,
    color: "#6b7280",
    marginTop: 2,
  },
  changeLink: {
    marginBottom: 16,
    marginTop: 4,
  },
  changeLinkText: {
    color: "#2563eb",
    fontSize: 14,
  },
  tipsCard: {
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    backgroundColor: "#f9fafb",
    padding: 12,
    marginBottom: 16,
  },
  tipsTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: "#374151",
    marginBottom: 8,
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  tipsColumns: {
    flexDirection: "row",
    gap: 8,
  },
  tipsCol: {
    flex: 1,
    gap: 4,
  },
  tipsDivider: {
    width: 1,
    backgroundColor: "#e5e7eb",
  },
  tipsGood: {
    fontSize: 12,
    fontWeight: "700",
    color: "#16a34a",
    marginBottom: 2,
  },
  tipsBad: {
    fontSize: 12,
    fontWeight: "700",
    color: "#dc2626",
    marginBottom: 2,
  },
  tipItem: {
    fontSize: 12,
    color: "#6b7280",
    lineHeight: 16,
  },
  photoButtons: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 12,
  },
  secondaryButton: {
    flex: 1,
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#2563eb",
    alignItems: "center",
  },
  secondaryButtonText: {
    color: "#2563eb",
    fontWeight: "600",
    fontSize: 15,
  },
  primaryButton: {
    backgroundColor: "#2563eb",
    padding: 14,
    borderRadius: 10,
    alignItems: "center",
    marginTop: 12,
  },
  primaryButtonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 16,
  },
  textInput: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    borderRadius: 10,
    padding: 12,
    fontSize: 16,
    backgroundColor: "#fff",
  },
  buttonDisabled: { opacity: 0.5 },
  suggestLink: { alignItems: "center", paddingVertical: 14 },
  suggestLinkText: { color: "#6b7280", fontSize: 14 },
  thumbnailGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 12,
    marginBottom: 4,
  },
  thumbnailWrapper: {
    width: "30%",
    aspectRatio: 1,
  },
  thumbnail: {
    width: "100%",
    height: "100%",
    borderRadius: 8,
  },
  thumbnailRemove: {
    position: "absolute",
    top: 4,
    right: 4,
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: "rgba(0,0,0,0.6)",
    alignItems: "center",
    justifyContent: "center",
  },
  thumbnailRemoveText: {
    color: "#fff",
    fontSize: 10,
    fontWeight: "700",
  },
  preview: {
    width: "100%",
    height: 260,
    borderRadius: 10,
    marginTop: 12,
  },
  statusText: {
    fontSize: 16,
    color: "#6b7280",
  },
  doneIcon: {
    fontSize: 56,
    color: "#16a34a",
  },
  doneTitle: {
    fontSize: 24,
    fontWeight: "700",
  },
  doneSubtitle: {
    fontSize: 16,
    color: "#6b7280",
  },
});
