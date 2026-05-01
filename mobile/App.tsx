import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { StatusBar } from "expo-status-bar";

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

export default function App() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <NavigationContainer>
          <Stack.Navigator
            screenOptions={{
              headerShown: false,
            }}
          >
            <Stack.Screen name="Home" component={HomeScreen} />
            <Stack.Screen
              name="Identify"
              component={IdentifyScreen}
              options={{
                headerShown: true,
                title: "Identify a Route",
                headerBackTitle: "Home",
                headerStyle: { backgroundColor: "#fff" },
                headerTintColor: "#2563eb",
                headerTitleStyle: { fontWeight: "700", color: "#111827" },
                headerShadowVisible: false,
              }}
            />
            <Stack.Screen
              name="Submit"
              component={SubmitScreen}
              options={{
                headerShown: true,
                title: "Submit a Photo",
                headerBackTitle: "Home",
                headerStyle: { backgroundColor: "#fff" },
                headerTintColor: "#2563eb",
                headerTitleStyle: { fontWeight: "700", color: "#111827" },
                headerShadowVisible: false,
              }}
            />
            <Stack.Screen
              name="Review"
              component={ReviewScreen}
              options={{
                headerShown: true,
                title: "Review Submissions",
                headerBackTitle: "Home",
                headerStyle: { backgroundColor: "#fff" },
                headerTintColor: "#2563eb",
                headerTitleStyle: { fontWeight: "700", color: "#111827" },
                headerShadowVisible: false,
              }}
            />
          </Stack.Navigator>
        </NavigationContainer>
        <StatusBar style="auto" />
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
