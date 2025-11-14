package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	envAnthropicAPIKey = "ANTHROPIC_API_KEY"
	envAnthropicModel  = "ANTHROPIC_MODEL"
	envLLMAPIKey       = "LLM_API_KEY"
	envLLMModel        = "LLM_MODEL"

	defaultModel          = "claude-sonnet-4-5-20250929"
	defaultTimeoutSeconds = 120
	anthropicAPIVersion   = "2023-06-01"
	anthropicEndpoint     = "https://api.anthropic.com/v1/messages"
	anthropicBetaHeader   = "tools-2024-04-04"

	defaultLLMRetryAttempts  = 3
	defaultLLMRetryBaseDelay = 500 * time.Millisecond
	maxLLMErrorSnippetRunes  = 200
)

// Message описывает одну реплику в формате роль/контент.
type Message struct {
	Role    string
	Content string
}

// Tool описывает доступный инструмент для модели (аналог JSON schema).
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

// Request агрегирует данные для обращения к LLM.
type Request struct {
	SystemPrompt string
	Messages     []Message
	Tools        []Tool
	Temperature  float32
	MaxTokens    int
}

// Response содержит итоговый текст от модели.
type Response struct {
	Text string
}

// Client описывает интерфейс взаимодействия с LLM.
type Client interface {
	Name() string
	Generate(ctx context.Context, req Request) (Response, error)
}

// Config хранит параметры подключения к LLM.
type Config struct {
	APIKey string
	Model  string
}

// NewClientFromEnv создаёт клиента Anthropic Claude, используя переменные окружения.
func NewClientFromEnv() (Client, error) {
	cfg := Config{
		APIKey: selectFirstNonEmpty(
			os.Getenv(envAnthropicAPIKey),
			os.Getenv(envLLMAPIKey),
		),
		Model: selectFirstNonEmpty(
			os.Getenv(envAnthropicModel),
			os.Getenv(envLLMModel),
			defaultModel,
		),
	}

	if cfg.APIKey == "" {
		return nil, fmt.Errorf("не задан API-ключ (ожидалась переменная %q)", envAnthropicAPIKey)
	}

	httpClient := &http.Client{
		Timeout: defaultTimeoutSeconds * time.Second,
	}

	return &anthropicClient{
		cfg:        cfg,
		httpClient: httpClient,
	}, nil
}

type anthropicClient struct {
	cfg        Config
	httpClient *http.Client
}

func (c *anthropicClient) Name() string {
	return c.cfg.Model
}

// Generate отправляет запрос к Anthropic Messages API и возвращает текст ответа.
func (c *anthropicClient) Generate(ctx context.Context, req Request) (Response, error) {
	if len(req.Messages) == 0 {
		return Response{}, errors.New("нельзя вызвать LLM без сообщений")
	}

	// Формируем базовый payload под спецификацию Anthropic Messages API.
	payload := anthropicPayload{
		Model:       c.cfg.Model,
		MaxTokens:   max(req.MaxTokens, 1024),
		Temperature: float64(req.Temperature),
	}

	if req.SystemPrompt != "" {
		payload.System = req.SystemPrompt
	}

	// Перекладываем сообщения пользователя/агента в формат role/content, который ожидает API.
	payload.Messages = make([]anthropicMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		payload.Messages = append(payload.Messages, anthropicMessage{
			Role: m.Role,
			Content: []anthropicContent{
				{Type: "text", Text: m.Content},
			},
		})
	}

	if len(req.Tools) > 0 {
		// Если модель будет вызывать инструменты, прикладываем их описание и JSON-схемы.
		payload.Tools = make([]anthropicTool, 0, len(req.Tools))
		for _, tool := range req.Tools {
			payload.Tools = append(payload.Tools, anthropicTool(tool))
		}
	}

	// Сериализуем payload перед отправкой по сети.
	body, err := json.Marshal(payload)
	if err != nil {
		return Response{}, fmt.Errorf("не удалось сериализовать запрос к LLM: %w", err)
	}

	var lastErr error

	// Организуем несколько попыток, чтобы пережить временные ошибки сети или 5xx ответов.
	for attempt := 1; attempt <= defaultLLMRetryAttempts; attempt++ {
		retryRequired := false

		// Конструируем HTTP-запрос для текущей попытки.
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, anthropicEndpoint, bytes.NewReader(body))
		if err != nil {
			return Response{}, fmt.Errorf("не удалось создать HTTP-запрос к LLM: %w", err)
		}

		// Заполняем необходимые заголовки: тип контента, ключ, версия API и beta-флаг для tool calling.
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("x-api-key", c.cfg.APIKey)
		httpReq.Header.Set("anthropic-version", anthropicAPIVersion)
		if anthropicBetaHeader != "" {
			httpReq.Header.Set("anthropic-beta", anthropicBetaHeader)
		}

		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("ошибка обращения к LLM: %w", err)
			retryRequired = true
		} else {
			// Читаем тело ответа целиком, чтобы иметь возможность повторить попытку при сбоях.
			bodyBytes, readErr := io.ReadAll(resp.Body)
			resp.Body.Close()
			if readErr != nil {
				lastErr = fmt.Errorf("не удалось прочитать ответ LLM: %w", readErr)
				retryRequired = true
			} else if resp.StatusCode >= 500 {
				// 5xx ошибки считаем временными и пытаемся повторить запрос с экспоненциальной задержкой.
				snippet := truncateRunes(strings.TrimSpace(string(bodyBytes)), maxLLMErrorSnippetRunes)
				if snippet == "" {
					snippet = "пустой ответ"
				}
				lastErr = fmt.Errorf("LLM вернул код %d: %s", resp.StatusCode, snippet)
				retryRequired = true
			} else if resp.StatusCode >= 400 {
				// Для 4xx считаем, что ошибка фатальная (неправильный запрос или ограничения тарифа).
				var apiErr anthropicError
				if err := json.Unmarshal(bodyBytes, &apiErr); err != nil {
					return Response{}, fmt.Errorf("LLM вернул код %d и некорректное тело ответа", resp.StatusCode)
				}
				return Response{}, fmt.Errorf("LLM вернул код %d: %s", resp.StatusCode, apiErr.Error())
			} else {
				// Пытаемся разобрать успешный ответ и собрать итоговый текст из блоков.
				var apiResp anthropicResponse
				if err := json.Unmarshal(bodyBytes, &apiResp); err != nil {
					lastErr = fmt.Errorf("не удалось распарсить ответ LLM: %w", err)
					retryRequired = true
				} else {
					var builder bytes.Buffer
					for _, block := range apiResp.Content {
						if block.Type == "text" {
							builder.WriteString(block.Text)
						}
					}
					return Response{Text: builder.String()}, nil
				}
			}
		}

		if retryRequired && attempt < defaultLLMRetryAttempts {
			// Экспоненциально увеличиваем задержку между повторными попытками.
			time.Sleep(defaultLLMRetryBaseDelay * time.Duration(1<<(attempt-1)))
			continue
		}

		break
	}

	if lastErr != nil {
		// Если все попытки провалились, возвращаем последнюю значимую ошибку.
		return Response{}, lastErr
	}

	return Response{}, errors.New("LLM не вернул ответ")
}

type anthropicPayload struct {
	Model       string             `json:"model"`
	System      string             `json:"system,omitempty"`
	Messages    []anthropicMessage `json:"messages"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature float64            `json:"temperature"`
}

type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

type anthropicContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

type anthropicResponse struct {
	Content []anthropicContent `json:"content"`
}

type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func (e anthropicError) Error() string {
	if e.Message != "" {
		return e.Message
	}
	return e.Type
}

func selectFirstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func truncateRunes(text string, limit int) string {
	if limit <= 0 {
		return ""
	}
	runes := []rune(text)
	if len(runes) <= limit {
		return text
	}
	return string(runes[:limit]) + "..."
}
