#!/bin/bash

if test -z "$1"; then
  echo "Usage: $0 archive-file.mongodump.xz"
  exit 1
fi
ARCHIVE="$1"

xzcat "$ARCHIVE" | mongorestore --archive
