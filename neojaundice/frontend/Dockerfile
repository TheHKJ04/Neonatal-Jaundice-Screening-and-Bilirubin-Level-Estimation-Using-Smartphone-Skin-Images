# Multi-stage build: Node compiles the Vite app; Nginx serves static assets.
FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:1.28-alpine AS runtime
ENV API_UPSTREAM=http://api:8000
COPY nginx/default.conf.template /etc/nginx/default.conf.template
COPY docker/render-nginx-config.sh /docker-entrypoint.d/40-render-nginx-config.sh
RUN chmod 0755 /docker-entrypoint.d/40-render-nginx-config.sh
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
