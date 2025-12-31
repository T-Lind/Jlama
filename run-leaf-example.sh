#!/bin/bash
# Script to run the LEAF model example

cd "$(dirname "$0")/jlama-core" || exit 1

# Build if needed
if [ ! -f "target/jlama-core-0.9.0-beta.jar" ]; then
    echo "Building jlama-core..."
    mvn package -DskipTests || exit 1
fi

# Copy dependencies if needed
if [ ! -d "target/dependency" ] || [ -z "$(ls -A target/dependency/*.jar 2>/dev/null)" ]; then
    echo "Copying dependencies..."
    # Get absolute path to jlama-core directory before changing
    JLAMA_CORE_DIR="$(pwd)"
    cd ..
    # Use absolute path to avoid nested directory issues when running from parent
    mvn dependency:copy-dependencies -pl jlama-core -DoutputDirectory="$JLAMA_CORE_DIR/target/dependency" || exit 1
    cd "$JLAMA_CORE_DIR" || exit 1
fi

# Build classpath - explicitly add all dependency jars
CP="target/jlama-core-0.9.0-beta.jar"
DEPENDENCY_COUNT=0
if [ -d "target/dependency" ]; then
    # Add each jar explicitly to avoid wildcard expansion issues
    for jar in target/dependency/*.jar; do
        if [ -f "$jar" ]; then
            CP="$CP:$jar"
            DEPENDENCY_COUNT=$((DEPENDENCY_COUNT + 1))
        fi
    done
fi

if [ "$DEPENDENCY_COUNT" -eq 0 ]; then
    echo "ERROR: No dependency jars found in target/dependency/"
    exit 1
fi

echo "Found $DEPENDENCY_COUNT dependency jars"

# Run the example
echo "Running LEAF model example..."
echo "Classpath: $CP"
echo ""
java -cp "$CP" \
    --add-modules jdk.incubator.vector \
    --enable-preview \
    com.github.tjake.jlama.examples.LeafModelExample "$@"

