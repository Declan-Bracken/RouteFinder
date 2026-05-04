import React, { useEffect, useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { useNavigation, useFocusEffect } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import { RootStackParamList } from "../../App";
import { getPendingImages } from "../api/client";

type Nav = StackNavigationProp<RootStackParamList, "Home">;

export default function HomeScreen() {
  const nav = useNavigation<Nav>();
  const [pendingCount, setPendingCount] = useState<number | null>(null);

  useFocusEffect(
    React.useCallback(() => {
      getPendingImages()
        .then(({ count }) => setPendingCount(count))  // count now includes areas + routes
        .catch(() => setPendingCount(null));
    }, []),
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>RouteFinder</Text>
      <Text style={styles.subtitle}>What would you like to do?</Text>

      <TouchableOpacity style={styles.card} onPress={() => nav.navigate("Identify")}>
        <Text style={styles.cardIcon}>🔍</Text>
        <View style={styles.cardText}>
          <Text style={styles.cardTitle}>Identify a route</Text>
          <Text style={styles.cardDesc}>Take a photo and find out what route it is</Text>
        </View>
      </TouchableOpacity>

      <TouchableOpacity style={styles.card} onPress={() => nav.navigate("Submit")}>
        <Text style={styles.cardIcon}>📸</Text>
        <View style={styles.cardText}>
          <Text style={styles.cardTitle}>Submit a photo</Text>
          <Text style={styles.cardDesc}>Tag and upload a photo for a known route</Text>
        </View>
      </TouchableOpacity>

      <TouchableOpacity style={styles.card} onPress={() => nav.navigate("Review")}>
        <Text style={styles.cardIcon}>✅</Text>
        <View style={styles.cardText}>
          <Text style={styles.cardTitle}>
            Review submissions
            {pendingCount !== null && pendingCount > 0 && (
              <Text style={styles.badge}>  {pendingCount} pending</Text>
            )}
          </Text>
          <Text style={styles.cardDesc}>Approve or reject submitted photos</Text>
        </View>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    paddingHorizontal: 24,
    paddingTop: 32,
    gap: 16,
  },
  title: { fontSize: 32, fontWeight: "800", marginBottom: 4 },
  subtitle: { fontSize: 16, color: "#6b7280", marginBottom: 16 },
  card: {
    flexDirection: "row",
    alignItems: "center",
    gap: 16,
    padding: 20,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    backgroundColor: "#f9fafb",
  },
  cardIcon: { fontSize: 32 },
  cardText: { flex: 1 },
  cardTitle: { fontSize: 17, fontWeight: "700" },
  cardDesc: { fontSize: 14, color: "#6b7280", marginTop: 2 },
  badge: { fontSize: 13, fontWeight: "600", color: "#2563eb" },
});
