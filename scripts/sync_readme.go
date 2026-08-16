package main

import (
	"bufio"
	"bytes"
	"fmt"

	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"unicode/utf16"
)

const (
	readmePath        = "README.md"
	quickstartPath    = "examples/quickstart/main.go"
	gitignorePath     = ".gitignore"
	treeStartMarker   = "<!-- START_PROJECT_TREE -->"
	treeEndMarker     = "<!-- END_PROJECT_TREE -->"
	codeStartMarker   = "<!-- START_QUICKSTART_CODE -->"
	codeEndMarker     = "<!-- END_QUICKSTART_CODE -->"
	outputStartMarker = "<!-- START_XOR_OUTPUT -->"
	outputEndMarker   = "<!-- END_XOR_OUTPUT -->"
)

func main() {
	readmeBytes, err := os.ReadFile(readmePath)
	if err != nil {
		fmt.Printf("Error reading %s: %v\n", readmePath, err)
		os.Exit(1)
	}
	readmeText := decodeToUTF8(readmeBytes)

	quickstartBytes, err := os.ReadFile(quickstartPath)
	if err != nil {
		fmt.Printf("Error reading %s: %v\n", quickstartPath, err)
		os.Exit(1)
	}
	quickstartText := decodeToUTF8(quickstartBytes)

	// Execute XOR demo to capture exact terminal output
	cmd := exec.Command("go", "run", "./examples/xor/main.go")
	outputBytes, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("Error running XOR demo: %v\nOutput: %s\n", err, decodeToUTF8(outputBytes))
		os.Exit(1)
	}
	outputText := decodeToUTF8(outputBytes)

	// Read and parse .gitignore dynamically
	gitignorePatterns := parseGitignore(gitignorePath)

	// 1. Generate ASCII Directory Tree dynamically
	treeText := generateDirectoryTree(".", gitignorePatterns)
	treeBlock := fmt.Sprintf("%s\n```text\n%s```\n%s", treeStartMarker, treeText, treeEndMarker)
	updatedText, err := replaceBetweenMarkers(readmeText, treeStartMarker, treeEndMarker, treeBlock)
	if err != nil {
		fmt.Printf("Error updating directory tree: %v\n", err)
		os.Exit(1)
	}

	// 2. Inject quickstart code snippet
	codeBlock := fmt.Sprintf("%s\n```go\n%s```\n%s", codeStartMarker, quickstartText, codeEndMarker)
	updatedText, err = replaceBetweenMarkers(updatedText, codeStartMarker, codeEndMarker, codeBlock)
	if err != nil {
		fmt.Printf("Error updating quickstart code: %v\n", err)
		os.Exit(1)
	}

	// 3. Inject XOR execution output
	outputBlock := fmt.Sprintf("%s\n```text\n%s\n```\n%s", outputStartMarker, strings.TrimSpace(outputText), outputEndMarker)
	updatedText, err = replaceBetweenMarkers(updatedText, outputStartMarker, outputEndMarker, outputBlock)
	if err != nil {
		fmt.Printf("Error updating XOR output: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile(readmePath, []byte(updatedText), 0644); err != nil {
		fmt.Printf("Error writing updated %s: %v\n", readmePath, err)
		os.Exit(1)
	}

	fmt.Println("README.md successfully synchronized with latest tree, code, and XOR execution output!")
}

// decodeToUTF8 automatically detects UTF-16 LE/BE Byte Order Marks (BOM) or UTF-16 null-byte patterns
// commonly produced by Windows PowerShell file redirection, converting them safely to UTF-8 strings.
func decodeToUTF8(b []byte) string {
	if len(b) >= 2 && b[0] == 0xFF && b[1] == 0xFE {
		// UTF-16 LE BOM
		b = b[2:]
		u16s := make([]uint16, len(b)/2)
		for i := 0; i < len(u16s); i++ {
			u16s[i] = uint16(b[2*i]) | (uint16(b[2*i+1]) << 8)
		}
		return string(utf16.Decode(u16s))
	} else if len(b) >= 2 && b[0] == 0xFE && b[1] == 0xFF {
		// UTF-16 BE BOM
		b = b[2:]
		u16s := make([]uint16, len(b)/2)
		for i := 0; i < len(u16s); i++ {
			u16s[i] = uint16(b[2*i+1]) | (uint16(b[2*i]) << 8)
		}
		return string(utf16.Decode(u16s))
	} else if len(b) >= 4 && b[1] == 0x00 && b[3] == 0x00 {
		// Heuristic detection: UTF-16 LE without BOM
		u16s := make([]uint16, len(b)/2)
		for i := 0; i < len(u16s); i++ {
			u16s[i] = uint16(b[2*i]) | (uint16(b[2*i+1]) << 8)
		}
		return string(utf16.Decode(u16s))
	}
	return string(bytes.TrimPrefix(b, []byte{0xEF, 0xBB, 0xBF})) // Strip UTF-8 BOM if present
}

// parseGitignore reads .gitignore lines dynamically without hardcoded patterns.
func parseGitignore(path string) []string {
	patterns := []string{".git"} // Always ignore .git directory
	file, err := os.Open(path)
	if err != nil {
		return patterns
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "/")
		line = strings.TrimSuffix(line, "/")
		patterns = append(patterns, line)
	}
	return patterns
}

func isIgnored(name string, patterns []string) bool {
	for _, p := range patterns {
		if matched, _ := filepath.Match(p, name); matched {
			return true
		}
		if name == p || strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}

func generateDirectoryTree(rootDir string, gitignorePatterns []string) string {
	var sb strings.Builder
	sb.WriteString("go-transformer-engine/\n")
	buildTree(rootDir, "", gitignorePatterns, &sb)
	return sb.String()
}

func buildTree(dirPath string, prefix string, gitignorePatterns []string, sb *strings.Builder) {
	entries, err := os.ReadDir(dirPath)
	if err != nil {
		return
	}

	var filtered []os.DirEntry
	for _, entry := range entries {
		name := entry.Name()
		if isIgnored(name, gitignorePatterns) {
			continue
		}
		filtered = append(filtered, entry)
	}

	for i, entry := range filtered {
		isLast := (i == len(filtered)-1)
		connector := "├── "
		childPrefix := prefix + "│   "
		if isLast {
			connector = "└── "
			childPrefix = prefix + "    "
		}

		name := entry.Name()
		if entry.IsDir() {
			name += "/"
		}
		sb.WriteString(prefix + connector + name + "\n")

		if entry.IsDir() {
			buildTree(filepath.Join(dirPath, entry.Name()), childPrefix, gitignorePatterns, sb)
		}
	}
}

func replaceBetweenMarkers(fullText, startMarker, endMarker, replacement string) (string, error) {
	startIdx := strings.Index(fullText, startMarker)
	if startIdx == -1 {
		return "", fmt.Errorf("marker %s not found in README.md", startMarker)
	}

	endIdx := strings.Index(fullText, endMarker)
	if endIdx == -1 {
		return "", fmt.Errorf("marker %s not found in README.md", endMarker)
	}

	if endIdx < startIdx {
		return "", fmt.Errorf("marker %s appears before %s", endMarker, startMarker)
	}

	endIdx += len(endMarker)

	return fullText[:startIdx] + replacement + fullText[endIdx:], nil
}
