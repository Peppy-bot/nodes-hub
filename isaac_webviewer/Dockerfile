# ----------------------------------------------------------------------
# Build stage
# ----------------------------------------------------------------------
FROM nvcr.io/nvidia/base/ubuntu:noble-20251013 AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
    && curl -fsSL --proto '=https' https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Scope the NVIDIA npm registry, scaffold the WebRTC viewer,
# then remove the temporary npm registry configuration.
RUN echo '@nvidia:registry=https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/' \
        >> /root/.npmrc \
    && npx @nvidia/create-ov-web-rtc-app@1.14.2 \
        --name web-viewer \
        --sample local-sample \
        --outputDir /app \
    && rm -f /root/.npmrc

WORKDIR /app/web-viewer

# ----------------------------------------------------------------------
# Make the viewer connect back to the host serving the webpage.
#
# This avoids baking a machine-specific IP address into the image.
#
# Browser:
#   http://<host>:8210
#
# WebRTC:
#   signalling -> <host>:49100
#   media      -> <host>:47998
# ----------------------------------------------------------------------
RUN sed -i \
        "s/signalingServer: [^,]*/signalingServer: window.location.hostname/" \
        src/main.ts \
    && sed -i \
        "/signalingServer:/a\    signalingPort: 49100,\n    mediaServer: window.location.hostname,\n    mediaPort: 47998,\n    forceWSS: false," \
        src/main.ts

# ----------------------------------------------------------------------
# Preserve useful browser shortcuts instead of forwarding them
# into the Isaac Sim stream.
#
# F11:
#   Browser fullscreen toggle.
#
# F5 / Ctrl+R / Cmd+R:
#   Refresh the viewer page.
#
# F12 / common DevTools shortcuts:
#   Prevent forwarding into the stream.
#
# F7:
#   Prevent Chrome caret-browsing prompt.
#
# Clipboard shortcuts are intentionally left untouched so the
# NVIDIA streaming library can handle them.
# ----------------------------------------------------------------------
RUN sed -i '/void app.initialize/i\
const sc = document.getElementById("stream-container");\
if (sc) {\
    sc.addEventListener("keydown", (e: KeyboardEvent) => {\
        if (e.key === "F11" || e.code === "F11") {\
            e.preventDefault();\
            e.stopPropagation();\
            if (document.fullscreenElement) {\
                document.exitFullscreen();\
            } else {\
                document.documentElement.requestFullscreen();\
            }\
            return;\
        }\
        if (e.key === "F7") {\
            e.preventDefault();\
            return;\
        }\
        if (e.key === "F5" || e.code === "F5") {\
            e.preventDefault();\
            e.stopPropagation();\
            window.location.reload();\
            return;\
        }\
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "r") {\
            e.preventDefault();\
            e.stopPropagation();\
            window.location.reload();\
            return;\
        }\
        if (e.key === "F12" || e.code === "F12") {\
            e.stopPropagation();\
            return;\
        }\
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && ["i", "j", "c"].includes(e.key.toLowerCase())) {\
            e.stopPropagation();\
            return;\
        }\
        if (e.metaKey && e.altKey && ["i", "j", "c"].includes(e.key.toLowerCase())) {\
            e.stopPropagation();\
            return;\
        }\
    }, true);\
}' src/main.ts

RUN npm install --ignore-scripts

RUN npm run build


# ----------------------------------------------------------------------
# Production stage
# ----------------------------------------------------------------------
FROM nvcr.io/nvidia/base/ubuntu:noble-20251013

# Python is installed explicitly because this image is also used as the
# base for the Peppy webviewer node, whose runscript launches a small
# Python NodeBuilder wrapper.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        python3 \
    && curl -fsSL --proto '=https' https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user.
RUN groupadd -r appuser --gid 1234 \
    && useradd \
        --no-log-init \
        -r \
        -g appuser \
        --uid 1234 \
        -d /app \
        -s /usr/sbin/nologin \
        appuser

WORKDIR /app

# Install only the dependencies required by the generated viewer.
COPY --from=builder --chmod=755 \
    /app/web-viewer/package.json \
    ./

RUN echo '@nvidia:registry=https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/' \
        >> /root/.npmrc \
    && npm install --ignore-scripts \
    && npm cache clean --force \
    && rm -f /root/.npmrc \
    && chown -R appuser:appuser /app

# Copy the production frontend generated in the build stage.
COPY --from=builder --chmod=755 \
    /app/web-viewer/dist \
    ./dist

USER appuser

EXPOSE 8210

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD curl -f http://localhost:8210/ || exit 1

# Allows the Docker image to work independently of Peppy as well.
# The Peppy Apptainer %runscript will override this execution path
# and invoke the Python NodeBuilder wrapper instead.
CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "8210"]
