import React, { useState, useCallback } from "react";
import {
  View,
  Text,
  TextInput,
  FlatList,
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
import { searchAreas, getRoutes, submitImage, Area, Route } from "../api/client";

type Step = "area" | "route" | "photo" | "submitting" | "done";

export default function SubmitScreen() {
  const [step, setStep] = useState<Step>("area");

  const [areaQuery, setAreaQuery] = useState("");
  const [areaSuggestions, setAreaSuggestions] = useState<Area[]>([]);
  const [selectedArea, setSelectedArea] = useState<Area | null>(null);
  const [areaLoading, setAreaLoading] = useState(false);

  const [routes, setRoutes] = useState<Route[]>([]);
  const [selectedRoute, setSelectedRoute] = useState<Route | null>(null);
  const [routesLoading, setRoutesLoading] = useState(false);

  const [imageUri, setImageUri] = useState<string | null>(null);

  const handleAreaChange = useCallback(async (text: string) => {
    setAreaQuery(text);
    setSelectedArea(null);
    if (text.length < 2) {
      setAreaSuggestions([]);
      return;
    }
    setAreaLoading(true);
    try {
      const results = await searchAreas(text);
      setAreaSuggestions(results);
    } catch {
      setAreaSuggestions([]);
    } finally {
      setAreaLoading(false);
    }
  }, []);

  const selectArea = useCallback(async (area: Area) => {
    setSelectedArea(area);
    setAreaQuery(area.full_path);
    setAreaSuggestions([]);
    setRoutesLoading(true);
    try {
      const r = await getRoutes(area.id);
      setRoutes(r);
      setStep("route");
    } catch {
      Alert.alert("Error", "Failed to load routes for this area.");
    } finally {
      setRoutesLoading(false);
    }
  }, []);

  const selectRoute = useCallback((route: Route) => {
    setSelectedRoute(route);
    setStep("photo");
  }, []);

  const pickImage = useCallback(async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
    });
    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
    }
  }, []);

  const takePhoto = useCallback(async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission needed", "Camera access is required to take photos.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      quality: 0.9,
    });
    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
    }
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
    setStep("area");
    setAreaQuery("");
    setAreaSuggestions([]);
    setSelectedArea(null);
    setRoutes([]);
    setSelectedRoute(null);
    setImageUri(null);
  }, []);

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
          {selectedRoute?.name} · {selectedRoute?.grade}
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
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <Text style={styles.heading}>Submit a route photo</Text>

        {/* Step 1: Area */}
        <Text style={styles.label}>Area</Text>
        <TextInput
          style={styles.input}
          placeholder="Search crag or area…"
          value={areaQuery}
          onChangeText={handleAreaChange}
          autoCorrect={false}
        />
        {areaLoading && <ActivityIndicator style={{ marginBottom: 8 }} />}
        {areaSuggestions.length > 0 && (
          <View style={styles.dropdown}>
            {areaSuggestions.map((a) => (
              <TouchableOpacity
                key={a.id}
                style={styles.dropdownItem}
                onPress={() => selectArea(a)}
              >
                <Text style={styles.dropdownText}>{a.full_path}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {/* Step 2: Route */}
        {(step === "route" || step === "photo") && (
          <>
            <Text style={styles.label}>Route</Text>
            {routesLoading ? (
              <ActivityIndicator style={{ marginBottom: 8 }} />
            ) : (
              <View style={styles.routeList}>
                {routes.map((r) => (
                  <TouchableOpacity
                    key={r.id}
                    style={[
                      styles.routeItem,
                      selectedRoute?.id === r.id && styles.routeItemSelected,
                    ]}
                    onPress={() => selectRoute(r)}
                  >
                    <Text style={styles.routeName}>{r.name}</Text>
                    <Text style={styles.routeGrade}>{r.grade}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            )}
          </>
        )}

        {/* Step 3: Photo */}
        {step === "photo" && selectedRoute && (
          <>
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
    marginTop: 2,
    overflow: "hidden",
  },
  dropdownItem: {
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#f3f4f6",
  },
  dropdownText: {
    fontSize: 15,
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
  routeItemSelected: {
    borderColor: "#2563eb",
    backgroundColor: "#eff6ff",
  },
  routeName: {
    fontSize: 15,
    flex: 1,
  },
  routeGrade: {
    fontSize: 14,
    fontWeight: "600",
    color: "#6b7280",
    marginLeft: 8,
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
