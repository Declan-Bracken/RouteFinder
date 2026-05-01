/**
 * Shared area + route search dropdown.
 * Used by SubmitScreen (area → routes → photo) and IdentifyScreen (area constraint).
 *
 * Parent owns the query value/state; this component handles debouncing,
 * API calls, dropdown rendering, and route-count warnings/limits.
 */
import React, { useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TextInput,
  ActivityIndicator,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
} from "react-native";
import { unifiedSearch, AreaResult, RouteResult, SearchResults } from "../api/client";

interface Props {
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  showRoutes?: boolean;        // false = areas only (Identify). default true
  routeCountWarn?: number;     // show yellow warning above this
  routeCountLimit?: number;    // block selection above this
  onSelectArea: (area: AreaResult) => void;
  onSelectRoute?: (route: RouteResult) => void;
}

export default function AreaRouteSearch({
  value,
  onChangeText,
  placeholder = "Search area or route…",
  showRoutes = true,
  routeCountWarn,
  routeCountLimit,
  onSelectArea,
  onSelectRoute,
}: Props) {
  const [results, setResults] = useState<SearchResults | null>(null);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (value.length < 2) { setResults(null); return; }

    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        setResults(await unifiedSearch(value));
      } catch {
        setResults(null);
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [value]);

  const handleSelectArea = (area: AreaResult) => {
    if (routeCountLimit && area.route_count > routeCountLimit) {
      Alert.alert(
        "Area too large",
        `${area.name} has ${area.route_count} routes. Select a more specific area for reliable results.`,
      );
      return;
    }
    setResults(null);
    onSelectArea(area);
  };

  const handleSelectRoute = (route: RouteResult) => {
    setResults(null);
    onSelectRoute?.(route);
  };

  const areas  = results?.areas  ?? [];
  const routes = showRoutes ? (results?.routes ?? []) : [];
  const hasResults = areas.length > 0 || routes.length > 0;

  return (
    <View>
      <TextInput
        style={styles.input}
        placeholder={placeholder}
        value={value}
        onChangeText={onChangeText}
        autoCorrect={false}
      />

      {loading && <ActivityIndicator style={{ marginTop: 8 }} />}

      {hasResults && (
        <ScrollView
          style={styles.dropdown}
          keyboardShouldPersistTaps="handled"
          nestedScrollEnabled
        >
          {areas.length > 0 && (
            <View>
              {showRoutes && (
                <Text style={styles.sectionHeader}>Areas</Text>
              )}
              {areas.map((area) => {
                const blocked = !!(routeCountLimit && area.route_count > routeCountLimit);
                const warned  = !!(routeCountWarn  && area.route_count > routeCountWarn && !blocked);
                return (
                  <TouchableOpacity
                    key={area.id}
                    style={[styles.item, blocked && styles.itemBlocked]}
                    onPress={() => handleSelectArea(area)}
                  >
                    <Text style={[styles.itemPrimary, blocked && styles.textMuted]}>
                      {area.full_path}
                    </Text>
                    <Text style={[styles.itemMeta, blocked && styles.textRed, warned && styles.textAmber]}>
                      {area.route_count} routes
                      {blocked ? " — too large" : warned ? " — large area" : ""}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          )}

          {routes.length > 0 && (
            <View>
              <Text style={styles.sectionHeader}>Routes</Text>
              {routes.map((route) => (
                <TouchableOpacity
                  key={route.id}
                  style={styles.item}
                  onPress={() => handleSelectRoute(route)}
                >
                  <Text style={styles.itemPrimary}>{route.name}</Text>
                  <Text style={styles.itemMeta}>
                    {route.grade ? `${route.grade} · ` : ""}{route.area}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
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
  item: {
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: "#f3f4f6",
  },
  itemBlocked: { backgroundColor: "#fafafa" },
  itemPrimary: { fontSize: 15, fontWeight: "500" },
  itemMeta: { fontSize: 13, color: "#6b7280", marginTop: 2 },
  textMuted: { color: "#9ca3af" },
  textRed: { color: "#dc2626" },
  textAmber: { color: "#d97706" },
});
