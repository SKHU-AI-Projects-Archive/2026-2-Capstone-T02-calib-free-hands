plugins {
    id("com.android.application")
}

android {
    namespace = "com.camchk"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.camchk"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    // Library used as-is (no source changes).
    implementation("com.github.pedroSG94.RootEncoder:library:2.8.2")
    // The "android" variant avoids java.beans classes that do not exist on Android.
    implementation("org.yaml:snakeyaml:1.33:android")
    // QR scanner UI from Google Play services (no camera permission needed): "Scan QR" sets the stream URL.
    implementation("com.google.android.gms:play-services-code-scanner:16.1.0")
}
