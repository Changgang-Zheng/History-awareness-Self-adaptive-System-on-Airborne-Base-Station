#!/bin/bash

BASEDIR="$(dirname "$0")"

DQN_DB=DQN_Data_Base
DQN_DIR="$BASEDIR/DQN-changgang"

QL_DB=Q_Learning_Data_Base
QL_DIR="$BASEDIR/Q-Learning-changgang"

SARSA_DB=SARSA_Data_Base
SARSA_DIR="$BASEDIR/SARSA_changgang"

EPISODES=100
STEPS=20000

# Fewer steps for DQN (20000 steps takes far too long)
STEPS_DQN=2000

# runExperiment(dbName, scriptDir)
runExperiment() {
    DB_NAME="$1"
    SCRIPT_DIR="$2"
    N_STEPS="$3"

    mongo "$DB_NAME" --eval "db.dropDatabase()" || true
    pushd "$SCRIPT_DIR"
    python main.py --database_name "$DB_NAME" --episode "$EPISODES" --step "$N_STEPS"
    popd
}

# Uncomment to run in sequence
runExperiment "$DQN_DB" "$DQN_DIR" "$STEPS_DQN" \
    & runExperiment "$QL_DB" "$QL_DIR" "$STEPS" \
    & runExperiment "$SARSA_DB" "$SARSA_DIR" "$STEPS"

# Uncomment to run in parallel
# runExperiment "$DQN_DB" "$DQN_DIR" "$STEPS_DQN" \
#     & runExperiment "$QL_DB" "$QL_DIR" "$STEPS" \
#     & runExperiment "$SARSA_DB" "$SARSA_DIR" "$STEPS"
