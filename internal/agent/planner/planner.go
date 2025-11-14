package planner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/polzovatel/ai-agent-for-browser/internal/llm"
	"github.com/polzovatel/ai-agent-for-browser/internal/tools"
)

const (
	systemPrompt = `Ты — автономный агент, который решает задачи в реальном браузере.
На каждом шаге ты должен:
1. Проанализировать доступную историю действий и текущую цель.
2. Решить, какое следующее действие предпринять, используя доступные инструменты.
3. Всегда отвечать строго в формате JSON, без дополнительных комментариев.

Основные правила работы:
- Начинай с общего обзора страницы (read_page, collect_texts), чтобы понять структуру интерфейса.
- Комбинируй разные подходы к поиску элементов: текст, роль, селектор, позиция в DOM.
- Перед кликом убеждайся, что элемент видим: при необходимости прокручивай страницу или уточняй селектор.
- Если элемент повторяется, сначала уточни структуру области (например, через collect_texts) и выбирай конкретные селекторы: уточняющие псевдоклассы, data-* атрибуты или комбинацию родитель+элемент.
- Когда нужно собрать список ссылок, сначала пробуй инструмент collect_links или указывай attribute=href, прежде чем переключаться на другие атрибуты.
- Когда получаешь hash-ссылки или относительные адреса, можешь открыть их через navigate с полной ссылкой текущего сайта.
- Перед кликом по контрольным зонам убеждайся, что селектор относится к рабочей области страницы, а не к навигации или боковым панелям.
- Если клик по видимой надписи ничего не делает, предположи, что это статический текст: ищи соседние элементы с ролями (button, link), проверяй атрибуты или выбирай ближайший контейнер, который действительно реагирует на взаимодействие.
- Если инструмент возвращает пустой ответ, меняй стратегию: уточняй селектор, используй другой инструмент или комбинируй подходы вместо повторного вызова с теми же параметрами.
- Фиксируй промежуточные результаты через add_note и опирайся на заметки, чтобы не обрабатывать один и тот же объект повторно.
- При сомнительных или необратимых действиях запрашивай подтверждение у пользователя (request_user_input).
`

	envPlannerTemperature     = "ANTHROPIC_TEMPERATURE"
	defaultPlannerTemperature = 0.15
	minPlannerTemperature     = 0.0
	maxPlannerTemperature     = 1.0
)

var plannerTemperature = loadPlannerTemperature()

// Engine описывает возможности планировщика, которые нужны оркестратору.
type Engine interface {
	Name() string
	Next(ctx context.Context, state State) (Decision, error)
}

// HistoryItem фиксирует предыдущие шаги агента.
type HistoryItem struct {
	Step        int            `json:"step"`
	Thought     string         `json:"thought"`
	ActionName  string         `json:"action_name"`
	ActionInput map[string]any `json:"action_input,omitempty"`
	Observation string         `json:"observation"`
}

// State передаёт планировщику контекст текущего шага.
type State struct {
	Step            int
	TaskDescription string
	History         []HistoryItem
	Tools           []tools.Tool
}

// Decision описывает результат рассуждения планировщика для следующего шага агента.
type Decision struct {
	Thought      string
	ActionName   string
	ActionInput  map[string]any
	Finish       bool
	FinalSummary string
}

var (
	ErrPlannerResponse = errors.New("модель вернула некорректный ответ")
)

type planAndExecute struct {
	modelName string
	llm       llm.Client
}

// NewPlanAndExecute создаёт планировщик, использующий подсказки в стиле Plan-and-Execute.
func NewPlanAndExecute(client llm.Client) Engine {
	return &planAndExecute{
		modelName: client.Name(),
		llm:       client,
	}
}

func (p *planAndExecute) Name() string {
	return "plan-execute(" + p.modelName + ")"
}

func (p *planAndExecute) Next(ctx context.Context, state State) (Decision, error) {
	// Преобразуем описание инструментов в формат, который понимает LLM-клиент.
	llmTools := make([]llm.Tool, 0, len(state.Tools))
	for _, tool := range state.Tools {
		llmTools = append(llmTools, llm.Tool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		})
	}

	// Формируем payload для подсказки: задача, история, доступные инструменты, формат ожидаемого JSON.
	payload := map[string]any{
		"task":    state.TaskDescription,
		"step":    state.Step,
		"history": state.History,
		"tools":   state.Tools,
		"format": map[string]any{
			"description": "Ответ всегда в формате JSON. Пример структуры:",
			"schema": map[string]any{
				"thought":       "строка",
				"finish":        "bool",
				"final_summary": "строка, обязательна если finish=true",
				"action": map[string]any{
					"name":  "строка, имя инструмента",
					"input": "объект, параметры инструмента",
				},
			},
		},
	}

	// Преобразуем payload в JSON, чтобы подсунуть в prompt единой строкой.
	prompt, err := json.Marshal(payload)
	if err != nil {
		return Decision{}, fmt.Errorf("не удалось подготовить подсказку: %w", err)
	}

	// Собираем пользовательское сообщение: короткое описание текущего состояния и инструкция по формату.
	userMessage := fmt.Sprintf(`ТЕКУЩИЕ ДАННЫЕ:
%s

Инструкции:
- Сформулируй краткую мысль в поле "thought".
- Если нужно завершить задачу, установи "finish": true и опиши результат в "final_summary".
- Если требуются действия, укажи в "action" имя инструмента и параметры.
- Если инструмент не нужен, используй пустой объект { "name": "", "input": {} }.
- Не добавляй текст вне JSON.
`, string(prompt))

	// Запрашиваем ответ у LLM.
	response, err := p.llm.Generate(ctx, llm.Request{
		SystemPrompt: systemPrompt,
		Messages: []llm.Message{
			{Role: "user", Content: userMessage},
		},
		Tools:       llmTools,
		Temperature: plannerTemperature,
		MaxTokens:   1200,
	})
	if err != nil {
		return Decision{}, fmt.Errorf("ошибка планировщика: %w", err)
	}

	// Извлекаем JSON-объект из текста ответа.
	rawJSON, err := extractJSONObject(response.Text)
	if err != nil {
		return Decision{}, fmt.Errorf("%w: %v", ErrPlannerResponse, err)
	}

	// Парсим JSON в локальную структуру.
	var parsed struct {
		Thought      string `json:"thought"`
		Finish       bool   `json:"finish"`
		FinalSummary string `json:"final_summary"`
		Action       struct {
			Name  string         `json:"name"`
			Input map[string]any `json:"input"`
		} `json:"action"`
	}

	if err := json.Unmarshal([]byte(rawJSON), &parsed); err != nil {
		return Decision{}, fmt.Errorf("не удалось распарсить ответ планировщика: %w", err)
	}

	// Формируем итоговое решение для оркестратора.
	decision := Decision{
		Thought:      strings.TrimSpace(parsed.Thought),
		Finish:       parsed.Finish,
		FinalSummary: strings.TrimSpace(parsed.FinalSummary),
	}

	if parsed.Action.Input == nil {
		parsed.Action.Input = map[string]any{}
	}
	decision.ActionName = strings.TrimSpace(parsed.Action.Name)
	decision.ActionInput = parsed.Action.Input

	if decision.Finish && decision.FinalSummary == "" {
		return Decision{}, fmt.Errorf("%w: отсутствует final_summary при finish=true", ErrPlannerResponse)
	}

	return decision, nil
}

// extractJSONObject извлекает первый JSON-объект из произвольной строки.
func extractJSONObject(text string) (string, error) {
	depth := 0        // глубина вложенности фигурных скобок
	start := -1       // индекс начала JSON-объекта
	inString := false // находимся ли внутри строки
	escape := false   // обработка экранирования

	for i := 0; i < len(text); i++ {
		ch := text[i]

		if escape {
			escape = false
			continue
		}

		switch ch {
		case '\\':
			// если видим обратную косую внутри строки — следующий символ экранирован
			if inString {
				escape = true
			}
		case '"':
			// переключаем режим "внутри строки"
			inString = !inString
		case '{':
			if !inString {
				if depth == 0 {
					start = i
				}
				depth++
			}
		case '}':
			if !inString && depth > 0 {
				depth--
				if depth == 0 && start != -1 {
					return text[start : i+1], nil
				}
			}
		}
	}

	return "", fmt.Errorf("JSON object not found")
}

// loadPlannerTemperature читает температуру планировщика из переменной окружения.
func loadPlannerTemperature() float32 {
	raw := strings.TrimSpace(os.Getenv(envPlannerTemperature))
	if raw == "" {
		return float32(defaultPlannerTemperature)
	}
	value, err := strconv.ParseFloat(raw, 32)
	if err != nil {
		return float32(defaultPlannerTemperature)
	}
	if value < minPlannerTemperature {
		value = minPlannerTemperature
	}
	if value > maxPlannerTemperature {
		value = maxPlannerTemperature
	}
	return float32(value)
}

// SetTemperature позволяет переопределить температуру планировщика во время выполнения.
// Значение ограничивается допустимым диапазоном [0, 1].
func SetTemperature(value float32) {
	if value < minPlannerTemperature {
		value = minPlannerTemperature
	}
	if value > maxPlannerTemperature {
		value = maxPlannerTemperature
	}
	plannerTemperature = value
}
