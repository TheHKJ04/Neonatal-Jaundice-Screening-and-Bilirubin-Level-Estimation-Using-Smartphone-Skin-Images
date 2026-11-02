#!/bin/sh
set -eu

: "${API_UPSTREAM:=http://api:8000}"
export API_UPSTREAM

# Only substitute API_UPSTREAM; leave Nginx variables such as $host intact.
envsubst '${API_UPSTREAM}' \
  < /etc/nginx/default.conf.template \
  > /etc/nginx/conf.d/default.conf
