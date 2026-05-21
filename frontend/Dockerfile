# SemDisc frontend (Vite + Vue 3). Source is bind-mounted at runtime, so the
# image only carries node_modules. The dev server runs with polling enabled
# (see vite.config.js) for reliable hot reload over bind mounts.
FROM node:20-slim

WORKDIR /app

# Install deps first for layer caching. node_modules is kept inside the image
# (and shielded from the host bind mount by an anonymous volume in compose).
COPY package.json package-lock.json* ./
RUN npm install

EXPOSE 5173

# package.json "dev" already binds 0.0.0.0; keep port explicit for clarity.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
