package dflash

import (
	"strings"
	"testing"
)

func TestParseConfigRejectsUnsortedTargetLayers(t *testing.T) {
	data := []byte(`{
		"hidden_size": 256,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"head_dim": 64,
		"intermediate_size": 512,
		"vocab_size": 1000,
		"block_size": 8,
		"num_target_layers": 10,
		"dflash_config": {
			"mask_token_id": 1,
			"target_layer_ids": [5, 2, 8]
		}
	}`)
	_, err := parseConfig(data)
	if err == nil || !strings.Contains(err.Error(), "sorted") {
		t.Fatalf("err = %v, want sorted target_layer_ids error", err)
	}
}

func TestParseConfigRejectsMissingTargetLayers(t *testing.T) {
	data := []byte(`{
		"hidden_size": 256,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"head_dim": 64,
		"intermediate_size": 512,
		"vocab_size": 1000,
		"block_size": 8,
		"num_target_layers": 10,
		"dflash_config": { "mask_token_id": 1 }
	}`)
	_, err := parseConfig(data)
	if err == nil || !strings.Contains(err.Error(), "target_layer_ids") {
		t.Fatalf("err = %v, want target_layer_ids required", err)
	}
}

func TestParseConfigRejectsBadBlockSize(t *testing.T) {
	data := []byte(`{
		"hidden_size": 256,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"head_dim": 64,
		"intermediate_size": 512,
		"vocab_size": 1000,
		"block_size": 0,
		"num_target_layers": 10,
		"dflash_config": {
			"mask_token_id": 1,
			"target_layer_ids": [1]
		}
	}`)
	_, err := parseConfig(data)
	if err == nil || !strings.Contains(err.Error(), "block_size") {
		t.Fatalf("err = %v, want block_size error", err)
	}
}

func TestParseConfigMinimalOK(t *testing.T) {
	data := []byte(`{
		"hidden_size": 256,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"head_dim": 64,
		"intermediate_size": 512,
		"vocab_size": 1000,
		"block_size": 8,
		"num_target_layers": 10,
		"dflash_config": {
			"mask_token_id": 99,
			"target_layer_ids": [1, 4, 7]
		}
	}`)
	cfg, err := parseConfig(data)
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.BlockSizeValue != 8 || cfg.DFlash.MaskTokenID != 99 {
		t.Fatalf("cfg = %+v", cfg)
	}
	if len(cfg.LayerTypes) != 2 {
		t.Fatalf("default layer_types len = %d", len(cfg.LayerTypes))
	}
	if cfg.RopeTheta != 1000000 {
		t.Fatalf("default rope_theta = %v", cfg.RopeTheta)
	}
}

func TestParseConfigSlidingRequiresWindow(t *testing.T) {
	data := []byte(`{
		"hidden_size": 256,
		"num_hidden_layers": 1,
		"num_attention_heads": 4,
		"head_dim": 64,
		"intermediate_size": 512,
		"vocab_size": 1000,
		"block_size": 8,
		"num_target_layers": 10,
		"layer_types": ["sliding_attention"],
		"dflash_config": {
			"mask_token_id": 1,
			"target_layer_ids": [1]
		}
	}`)
	_, err := parseConfig(data)
	if err == nil || !strings.Contains(err.Error(), "sliding_window") {
		t.Fatalf("err = %v, want sliding_window error", err)
	}
}
