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
  SectionList,
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

  const reset = useCallback(() => {
    setStep("search");
    setQuery("");
    setSearchResults(null);
    setSelectedArea(null);
    setRoutes([]);
    setSelectedRoute(null);
    setImageUri(null);
  }, []);

  // Build SectionList sections from search results
  const sections = [];
  if (searchResults?.areas?.length) {
    sections.push({ title: "Areas", data: searchResults.areas, type: "area" as const });
  }
  if (searchResults?.routes?.length) {
    sections.push({ title: "Routes", data: searchResults.routes, type: "route" as const });
  }

  if (step === "submitting") {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#2563eb" />
        <Text style={styles.statusText}>Uploading…</Text>
      </View>
    );
  }

  if (step === "done") {
    return (
      <View style={styles.centered}>
        <Text style={styles.doneIcon}>✓</Text>
        <Text style={styles.doneTitle}>Submitted!</Text>
        <Text style={styles.doneSubtitle}>
          {selectedRoute?.name}
          {selectedRoute?.grade ? ` · ${selectedRoute.grade}` : ""}
        </Text>
        <TouchableOpacity style={styles.primaryButton} onPress={reset}>
          <Text style={styles.primaryButtonText}>Submit another</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView
        contentContainerStyle={styles.container}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={styles.heading}>Submit a route photo</Text>

        {/* Search box — always visible until photo step */}
        {step !== "photo" && (
          <>
            <Text style={styles.label}>
              {step === "routes" ? "Area" : "Search area or route"}
            </Text>
            <View style={styles.searchRow}>
              <TextInput
                style={[styles.input, { flex: 1 }]}
                placeholder="e.g. Mount Nemo, Lop Sided…"
                value={query}
                onChangeText={handleQueryChange}
                autoCorrect={false}
                editable={step === "search"}
              />
              {step === "routes" && (
                <TouchableOpacity style={styles.clearButton} onPress={reset}>
                  <Text style={styles.clearButtonText}>✕</Text>
                </TouchableOpacity>
              )}
            </View>
          </>
        )}

        {/* Search results dropdown */}
        {step === "search" && (
          <>
            {searchLoading && <ActivityIndicator style={{ marginTop: 8 }} />}
            {sections.length > 0 && (
              <View style={styles.dropdown}>
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
                          <Text style={styles.dropdownPrimary}>
                            {(item as AreaResult).full_path}
                          </Text>
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
                            {(item as RouteResult).grade
                              ? `${(item as RouteResult).grade} · `
                              : ""}
                            {(item as RouteResult).area}
                          </Text>
                        </TouchableOpacity>
                      )
                    )}
                  </View>
                ))}
              </View>
            )}
          </>
        )}

        {/* Route list after area selection */}
        {step === "routes" && (
          <>
            <Text style={styles.label}>Route</Text>
            {routesLoading ? (
              <ActivityIndicator style={{ marginTop: 8 }} />
            ) : (
              <View style={styles.routeList}>
                {routes.map((r) => (
                  <TouchableOpacity
                    key={r.id}
                    style={styles.routeItem}
                    onPress={() => selectRoute(r)}
                  >
                    <View style={{ flex: 1 }}>
                      <Text style={styles.routeName}>{r.name}</Text>
                      <Text style={styles.routeArea}>{r.area}</Text>
                    </View>
                    <Text style={styles.routeGrade}>{r.grade}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            )}
          </>
        )}

        {/* Photo step */}
        {step === "photo" && selectedRoute && (
          <>
            <Text style={styles.heading}>Submit a route photo</Text>
            <View style={styles.selectedRoute}>
              <Text style={styles.selectedRouteName}>{selectedRoute.name}</Text>
              <Text style={styles.selectedRouteMeta}>
                {selectedRoute.grade
                  ? `${selectedRoute.grade} · `
                  : ""}
                {"area" in selectedRoute ? selectedRoute.area : ""}
              </Text>
            </View>
            <TouchableOpacity style={styles.changeLink} onPress={reset}>
              <Text style={styles.changeLinkText}>Change route</Text>
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
          </>
        )}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    paddingBottom: 48,
  },
  centered: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: 16,
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
    marginTop: 16,
  },
  searchRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    borderRadius: 10,
    padding: 12,
    fontSize: 16,
    backgroundColor: "#fff",
  },
  clearButton: {
    padding: 10,
  },
  clearButtonText: {
    fontSize: 16,
    color: "#9ca3af",
  },
  dropdown: {
    borderWidth: 1,
    borderColor: "#d1d5db",
    borderRadius: 10,
    backgroundColor: "#fff",
    marginTop: 4,
    overflow: "hidden",
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
  routeList: {
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
  selectedRoute: {
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
    marginBottom: 8,
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
