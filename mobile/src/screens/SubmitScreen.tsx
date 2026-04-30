import React, { useState, useCallback, useRef } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import {
  unifiedSearch,
  getRoutes,
  submitImage,
  AreaResult,
  RouteResult,
  RouteDetail,
  SearchResults,
} from "../api/client";

type Step = "search" | "routes" | "photo" | "submitting" | "done";

export default function SubmitScreen() {
  const [step, setStep] = useState<Step>("search");

  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResults | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [selectedArea, setSelectedArea] = useState<AreaResult | null>(null);
  const [routes, setRoutes] = useState<RouteDetail[]>([]);
  const [routesLoading, setRoutesLoading] = useState(false);

  const [selectedRoute, setSelectedRoute] = useState<RouteDetail | RouteResult | null>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);

  const handleQueryChange = useCallback((text: string) => {
    setQuery(text);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (text.length < 2) {
      setSearchResults(null);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      setSearchLoading(true);
      try {
        const results = await unifiedSearch(text);
        setSearchResults(results);
      } catch {
        setSearchResults(null);
      } finally {
        setSearchLoading(false);
      }
    }, 300);
  }, []);

  const selectArea = useCallback(async (area: AreaResult) => {
    setSelectedArea(area);
    setSearchResults(null);
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
    setSearchResults(null);
    setStep("photo");
  }, []);

  // Go back to route list preserving area + routes state
  const backToRoutes = useCallback(() => {
    setSelectedRoute(null);
    setImageUri(null);
    setStep("routes");
  }, []);

  const resetAll = useCallback(() => {
    setStep("search");
    setQuery("");
    setSearchResults(null);
    setSelectedArea(null);
    setRoutes([]);
    setSelectedRoute(null);
    setImageUri(null);
  }, []);

  const pickImage = useCallback(async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
    });
    if (!result.canceled) setImageUri(result.assets[0].uri);
  }, []);

  const takePhoto = useCallback(async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission needed", "Camera access is required to take photos.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ quality: 0.9 });
    if (!result.canceled) setImageUri(result.assets[0].uri);
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!imageUri || !selectedRoute) return;
    setStep("submitting");
    try {
      await submitImage(imageUri, selectedRoute.id);
      setStep("done");
    } catch (e: any) {
      Alert.alert("Submit failed", e.message);
      setStep("photo");
    }
  }, [imageUri, selectedRoute]);

  // ── Submitting ──────────────────────────────────────────────────────────────
  if (step === "submitting") {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#2563eb" />
        <Text style={styles.statusText}>Uploading…</Text>
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
          <ScrollView style={styles.fill} contentContainerStyle={styles.routeListContent}>
            {routes.map((r) => (
              <TouchableOpacity key={r.id} style={styles.routeItem} onPress={() => selectRoute(r)}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.routeName}>{r.name}</Text>
                  <Text style={styles.routeArea}>{r.area}</Text>
                </View>
                <Text style={styles.routeGrade}>{r.grade}</Text>
              </TouchableOpacity>
            ))}
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
        <ScrollView contentContainerStyle={styles.container}>
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

          <Text style={styles.label}>Photo</Text>
          <View style={styles.photoButtons}>
            <TouchableOpacity style={styles.secondaryButton} onPress={takePhoto}>
              <Text style={styles.secondaryButtonText}>Take photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.secondaryButton} onPress={pickImage}>
              <Text style={styles.secondaryButtonText}>Choose from library</Text>
            </TouchableOpacity>
          </View>
          {imageUri && (
            <>
              <Image source={{ uri: imageUri }} style={styles.preview} resizeMode="cover" />
              <TouchableOpacity style={styles.primaryButton} onPress={handleSubmit}>
                <Text style={styles.primaryButtonText}>Submit</Text>
              </TouchableOpacity>
            </>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    );
  }

  // ── Search step ─────────────────────────────────────────────────────────────
  const sections: { title: string; data: (AreaResult | RouteResult)[]; type: "area" | "route" }[] = [];
  if (searchResults?.areas?.length) {
    sections.push({ title: "Areas", data: searchResults.areas, type: "area" });
  }
  if (searchResults?.routes?.length) {
    sections.push({ title: "Routes", data: searchResults.routes, type: "route" });
  }

  return (
    <KeyboardAvoidingView
      style={styles.fill}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <View style={styles.container}>
        <Text style={styles.heading}>Submit a route photo</Text>
        <Text style={styles.label}>Search area or route</Text>
        <TextInput
          style={styles.input}
          placeholder="e.g. Mount Nemo, Lop Sided…"
          value={query}
          onChangeText={handleQueryChange}
          autoCorrect={false}
        />

        {searchLoading && <ActivityIndicator style={{ marginTop: 8 }} />}

        {sections.length > 0 && (
          <ScrollView
            style={styles.dropdown}
            keyboardShouldPersistTaps="handled"
            nestedScrollEnabled
          >
            {sections.map((section) => (
              <View key={section.title}>
                <Text style={styles.sectionHeader}>{section.title}</Text>
                {section.data.map((item) =>
                  section.type === "area" ? (
                    <TouchableOpacity
                      key={(item as AreaResult).id}
                      style={styles.dropdownItem}
                      onPress={() => selectArea(item as AreaResult)}
                    >
                      <Text style={styles.dropdownPrimary}>{(item as AreaResult).full_path}</Text>
                      <Text style={styles.dropdownMeta}>
                        {(item as AreaResult).route_count} routes
                      </Text>
                    </TouchableOpacity>
                  ) : (
                    <TouchableOpacity
                      key={(item as RouteResult).id}
                      style={styles.dropdownItem}
                      onPress={() => selectRoute(item as RouteResult)}
                    >
                      <Text style={styles.dropdownPrimary}>{(item as RouteResult).name}</Text>
                      <Text style={styles.dropdownMeta}>
                        {(item as RouteResult).grade ? `${(item as RouteResult).grade} · ` : ""}
                        {(item as RouteResult).area}
                      </Text>
                    </TouchableOpacity>
                  )
                )}
              </View>
            ))}
          </ScrollView>
        )}
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
  input: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    borderRadius: 10,
    padding: 12,
    fontSize: 16,
    backgroundColor: "#fff",
  },
  dropdown: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    borderRadius: 10,
    backgroundColor: "#fff",
    marginTop: 4,
    maxHeight: 420,
  },
  sectionHeader: {
    fontSize: 11,
    fontWeight: "700",
    color: "#9ca3af",
    textTransform: "uppercase",
    letterSpacing: 0.5,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: "#f9fafb",
  },
  dropdownItem: {
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: "#f3f4f6",
  },
  dropdownPrimary: {
    fontSize: 15,
    fontWeight: "500",
  },
  dropdownMeta: {
    fontSize: 13,
    color: "#6b7280",
    marginTop: 2,
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
