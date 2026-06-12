package diffusiongemma

import (
	"iter"
)

// testConfig implements fs.Config for testing without GGUF files.
type testConfig map[string]any

func (c testConfig) Architecture() string { return "diffusiongemma" }

func (c testConfig) key(key string) string {
	switch {
	case len(key) >= len("tokenizer.") && key[:len("tokenizer.")] == "tokenizer.":
		return key
	case len(key) >= len("general.") && key[:len("general.")] == "general.":
		return key
	default:
		return "diffusiongemma." + key
	}
}

func (c testConfig) String(key string, defaultValue ...string) string {
	if v, ok := c[c.key(key)].(string); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return ""
}

func (c testConfig) Uint(key string, defaultValue ...uint32) uint32 {
	switch v := c[c.key(key)].(type) {
	case uint32:
		return v
	case int:
		return uint32(v)
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return 0
}

func (c testConfig) Float(key string, defaultValue ...float32) float32 {
	if v, ok := c[c.key(key)].(float32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return 0
}

func (c testConfig) Bool(key string, defaultValue ...bool) bool {
	if v, ok := c[c.key(key)].(bool); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return false
}

func (c testConfig) Strings(key string, defaultValue ...[]string) []string {
	if v, ok := c[c.key(key)].([]string); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Ints(key string, defaultValue ...[]int32) []int32 {
	if v, ok := c[c.key(key)].([]int32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Uints(key string, defaultValue ...[]uint32) []uint32 {
	if v, ok := c[c.key(key)].([]uint32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Floats(key string, defaultValue ...[]float32) []float32 {
	if v, ok := c[c.key(key)].([]float32); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Bools(key string, defaultValue ...[]bool) []bool {
	if v, ok := c[c.key(key)].([]bool); ok {
		return v
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return nil
}

func (c testConfig) Len() int { return len(c) }

func (c testConfig) Keys() iter.Seq[string] {
	return func(yield func(string) bool) {
		for key := range c {
			if !yield(key) {
				return
			}
		}
	}
}

func (c testConfig) Value(key string) any { return c[key] }
