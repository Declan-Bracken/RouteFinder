import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { StatusBar } from "expo-status-bar";
import { TouchableOpacity, Text } from "react-native";

import HomeScreen from "./src/screens/HomeScreen";
import SubmitScreen from "./src/screens/SubmitScreen";
import IdentifyScreen from "./src/screens/IdentifyScreen";
import ReviewScreen from "./src/screens/ReviewScreen";

export type RootStackParamList = {
  Home: undefined;
  Submit: undefined;
  Identify: undefined;
  Review: undefined;
};

const Stack = createStackNavigator<RootStackParamList>();

const sharedHeaderOptions = ({ navigation }: { navigation: any }) => ({
  headerShown: true,
  headerStyle: { backgroundColor: "#fff" },
  headerTitleStyle: { fontWeight: "700" as const, color: "#111827" },
  headerShadowVisible: true,
  headerLeft: () => (
    <TouchableOpacity
      onPress={() => navigation.goBack()}
      style={{ paddingHorizontal: 16, paddingVertical: 8 }}
    >
      <Text style={{ color: "#2563eb", fontSize: 16, fontWeight: "600" }}>← Back</Text>
    </TouchableOpacity>
  ),
});

export default function App() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <NavigationContainer>
          <Stack.Navigator screenOptions={{ headerShown: false }}>
            <Stack.Screen name="Home" component={HomeScreen} />
            <Stack.Screen
              name="Identify"
              component={IdentifyScreen}
              options={(props) => ({ ...sharedHeaderOptions(props), title: "Identify a Route" })}
            />
            <Stack.Screen
              name="Submit"
              component={SubmitScreen}
              options={(props) => ({ ...sharedHeaderOptions(props), title: "Submit a Photo" })}
            />
            <Stack.Screen
              name="Review"
              component={ReviewScreen}
              options={(props) => ({ ...sharedHeaderOptions(props), title: "Review Submissions" })}
            />
          </Stack.Navigator>
        </NavigationContainer>
        <StatusBar style="auto" />
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
