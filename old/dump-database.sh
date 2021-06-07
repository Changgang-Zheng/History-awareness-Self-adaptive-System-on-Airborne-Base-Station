#!/bin/bash

ARCHIVE_NAME="drones-$(date +'%Y%m%d%H%S').mongodump.xz"

mongodump --archive | xz -9 --stdout > "$ARCHIVE_NAME"
