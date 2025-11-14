package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/polzovatel/ai-agent-for-browser/internal/browser"
)

const (
	defaultReadPageLimit = 2000
)

// Toolbox описывает набор действий, доступных агенту.
type Toolbox interface {
	Describe() []Tool
	Invoke(ctx context.Context, name string, input map[string]any) (Result, error)
}

// Tool предоставляет метаданные об отдельных действиях.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

// Result хранит текстовое наблюдение, которое можно передать модели.
type Result struct {
	Observation string
}

// PromptFunc описывает функцию для запроса ввода пользователя.
type PromptFunc func(ctx context.Context, message string) (string, error)

type standardToolbox struct {
	controller browser.Controller
	tools      []Tool
	prompt     PromptFunc
}

// NewStandardToolbox привязывает базовый набор инструментов к переданному контроллеру.
func NewStandardToolbox(controller browser.Controller, prompt PromptFunc) Toolbox {
	return &standardToolbox{
		controller: controller,
		prompt:     prompt,
		tools: []Tool{
			{
				Name:        "navigate",
				Description: "Открыть указанную веб-страницу по URL.",
				InputSchema: map[string]any{
					"type":       "object",
					"required":   []string{"url"},
					"properties": map[string]any{"url": map[string]any{"type": "string", "description": "Полный URL страницы, например https://example.com"}},
				},
			},
			{
				Name:        "wait_for_element",
				Description: "Подождать появления элемента на странице.",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"selector"},
					"properties": map[string]any{
						"selector":   map[string]any{"type": "string", "description": "CSS-селектор целевого элемента"},
						"timeout_ms": map[string]any{"type": "integer", "description": "Тайм-аут ожидания в миллисекундах (по умолчанию 5000)"},
					},
				},
			},
			{
				Name:        "click",
				Description: "Нажать на элемент, указанный CSS-селектором или стратегией Playwright.",
				InputSchema: map[string]any{
					"type":       "object",
					"required":   []string{"selector"},
					"properties": map[string]any{"selector": map[string]any{"type": "string", "description": "Селектор для поиска элемента"}},
				},
			},
			{
				Name:        "click_role",
				Description: "Нажать на элемент по роли (ARIA) и имени, например кнопку или ссылку.",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"role"},
					"properties": map[string]any{
						"role":  map[string]any{"type": "string", "description": "Роль элемента (button, link, menuitem, checkbox, radio, option)"},
						"name":  map[string]any{"type": "string", "description": "Отображаемый текст или имя элемента"},
						"exact": map[string]any{"type": "boolean", "description": "Использовать точное совпадение имени"},
					},
				},
			},
			{
				Name:        "click_text",
				Description: "Нажать на элемент по видимому тексту.",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"text"},
					"properties": map[string]any{
						"text":  map[string]any{"type": "string", "description": "Видимый текст элемента, по которому нужно кликнуть"},
						"exact": map[string]any{"type": "boolean", "description": "Если true, текст должен совпадать в точности"},
					},
				},
			},
			{
				Name:        "fill",
				Description: "Заполнить поле ввода текстом.",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"selector", "text"},
					"properties": map[string]any{
						"selector": map[string]any{"type": "string", "description": "Селектор поля ввода"},
						"text":     map[string]any{"type": "string", "description": "Текст, который нужно ввести"},
						"submit":   map[string]any{"type": "boolean", "description": "Отправить форму после ввода (нажимая Enter)"},
					},
				},
			},
			{
				Name:        "read_page",
				Description: "Получить текстовую выжимку текущей страницы или выбранного элемента.",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{},
					"properties": map[string]any{
						"selector":  map[string]any{"type": "string", "description": "Необязательный селектор для чтения только части страницы"},
						"max_chars": map[string]any{"type": "integer", "description": "Ограничить количество символов в ответе"},
					},
				},
			},
			{
				Name:        "add_note",
				Description: "Сохранить краткую заметку в памяти агента.",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"text"},
					"properties": map[string]any{
						"text": map[string]any{
							"type":        "string",
							"description": "Текст заметки",
						},
					},
				},
			},
			{
				Name:        "press_key",
				Description: "Нажать клавишу на активном элементе или странице (например Enter).",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"key"},
					"properties": map[string]any{
						"key": map[string]any{"type": "string", "description": "Клавиша, совместимая с Playwright, например Enter или Control+L"},
					},
				},
			},
			{
				Name:        "hover",
				Description: "Навести курсор на элемент (можно использовать для появления подсказок, меню).",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"selector"},
					"properties": map[string]any{
						"selector": map[string]any{"type": "string", "description": "Селектор элемента, на который нужно навести курсор"},
					},
				},
			},
			{
				Name:        "collect_texts",
				Description: "Собрать тексты элементов по селектору, опционально указав атрибут.",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"selector"},
					"properties": map[string]any{
						"selector":  map[string]any{"type": "string", "description": "CSS-селектор искомых элементов"},
						"attribute": map[string]any{"type": "string", "description": "Имя атрибута вместо текста"},
						"limit":     map[string]any{"type": "integer", "description": "Максимальное количество элементов"},
					},
				},
			},
			{
				Name:        "collect_links",
				Description: "Собрать ссылки: текст и href для элементов по селектору.",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{},
					"properties": map[string]any{
						"selector": map[string]any{"type": "string", "description": "CSS-селектор ссылок (по умолчанию 'a')"},
						"limit":    map[string]any{"type": "integer", "description": "Максимальное количество ссылок"},
					},
				},
			},
			{
				Name:        "select_option",
				Description: "Выбрать значение в выпадающем списке по value, label или номеру.",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"selector"},
					"properties": map[string]any{
						"selector": map[string]any{"type": "string", "description": "Селектор элемента select"},
						"value":    map[string]any{"type": "string", "description": "Значение option (value)"},
						"label":    map[string]any{"type": "string", "description": "Текст option"},
						"index":    map[string]any{"type": "integer", "description": "Индекс option (0 — первый)"},
					},
				},
			},
			{
				Name:        "set_checkbox",
				Description: "Выставить состояние checkbox (true/false).",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"selector", "checked"},
					"properties": map[string]any{
						"selector": map[string]any{"type": "string", "description": "Селектор checkbox или контейнера"},
						"checked":  map[string]any{"type": "boolean", "description": "true — включить, false — отключить"},
					},
				},
			},
			{
				Name:        "scroll_page",
				Description: "Прокрутить страницу или элемент.",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{},
					"properties": map[string]any{
						"direction": map[string]any{"type": "string", "description": "Направление (down, up, top, bottom, page_down, page_up)"},
						"distance":  map[string]any{"type": "integer", "description": "Расстояние в пикселях (по умолчанию 600)"},
						"selector":  map[string]any{"type": "string", "description": "Если указать селектор, страница прокрутится до элемента"},
					},
				},
			},
			{
				Name:        "request_user_input",
				Description: "Запросить дополнительную информацию у пользователя (например, код подтверждения).",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"prompt"},
					"properties": map[string]any{
						"prompt": map[string]any{"type": "string", "description": "Текст вопроса пользователю"},
					},
				},
			},
			{
				Name:        "save_storage_state",
				Description: "Сохранить текущее состояние сессии браузера в указанный файл.",
				InputSchema: map[string]any{
					"type":     "object",
					"required": []string{"path"},
					"properties": map[string]any{
						"path": map[string]any{"type": "string", "description": "Путь к файлу для сохранения state"},
					},
				},
			},
		},
	}
}

func (t *standardToolbox) Describe() []Tool {
	return append([]Tool(nil), t.tools...)
}

func (t *standardToolbox) Invoke(ctx context.Context, name string, input map[string]any) (Result, error) {
	// Разв branch'иваем выполнение по имени инструмента и вызываем соответствующий метод контроллера.
	switch name {
	case "wait_for_element":
		selector, err := extractString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		timeoutMs := extractOptionalInt(input, "timeout_ms")
		if timeoutMs <= 0 {
			timeoutMs = 5000
		}
		if err := t.controller.WaitFor(ctx, selector, time.Duration(timeoutMs)*time.Millisecond); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("Элемент %q найден", selector)}, nil

	case "navigate":
		url, err := extractString(input, "url")
		if err != nil {
			return Result{}, err
		}
		if err := t.controller.Navigate(ctx, url); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("Открыта страница %s", url)}, nil

	case "click":
		selector, err := extractString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		if err := t.controller.Click(ctx, selector); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("Клик по элементу %q выполнен", selector)}, nil

	case "click_text":
		text, err := extractString(input, "text")
		if err != nil {
			return Result{}, err
		}
		exact := extractBool(input, "exact")
		if err := t.controller.ClickText(ctx, text, exact); err != nil {
			return Result{}, err
		}
		mode := "неточное"
		if exact {
			mode = "точное"
		}
		return Result{Observation: fmt.Sprintf("Клик по тексту %q выполнен (%s совпадение)", text, mode)}, nil

	case "click_role":
		role, err := extractString(input, "role")
		if err != nil {
			return Result{}, err
		}
		name := extractOptionalString(input, "name")
		exact := extractBool(input, "exact")
		if err := t.controller.ClickRole(ctx, role, name, exact); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("Клик по роли %q (%s) выполнен", role, name)}, nil

	case "fill":
		selector, err := extractString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		text, err := extractString(input, "text")
		if err != nil {
			return Result{}, err
		}
		submit := extractBool(input, "submit")
		if err := t.controller.Fill(ctx, selector, text); err != nil {
			return Result{}, err
		}
		if submit {
			if err := t.controller.Press(ctx, selector, "Enter"); err != nil {
				return Result{}, err
			}
		}
		return Result{Observation: fmt.Sprintf("Поле %q заполнено текстом %q", selector, text)}, nil

	case "read_page":
		selector := extractOptionalString(input, "selector")
		maxChars := extractOptionalInt(input, "max_chars")
		content, err := t.controller.Read(ctx, selector)
		if err != nil {
			return Result{}, err
		}
		limit := maxChars
		if limit <= 0 {
			limit = defaultReadPageLimit
		}
		if limit > 0 && len(content) > limit {
			content = content[:limit]
		}
		return Result{Observation: content}, nil

	case "collect_texts":
		selector, err := extractString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		attribute := extractOptionalString(input, "attribute")
		limit := extractOptionalInt(input, "limit")
		items, err := t.controller.CollectText(ctx, selector, attribute, limit)
		if err != nil {
			return Result{}, err
		}
		payload := map[string]any{"items": items}
		encoded, err := json.Marshal(payload)
		if err != nil {
			return Result{}, err
		}
		return Result{Observation: string(encoded)}, nil

	case "collect_links":
		selector := extractOptionalString(input, "selector")
		limit := extractOptionalInt(input, "limit")
		links, err := t.controller.CollectLinks(ctx, selector, limit)
		if err != nil {
			return Result{}, err
		}
		payload := map[string]any{"items": links}
		encoded, err := json.Marshal(payload)
		if err != nil {
			return Result{}, err
		}
		return Result{Observation: string(encoded)}, nil

	case "add_note":
		text, err := extractString(input, "text")
		if err != nil {
			return Result{}, err
		}
		note := strings.TrimSpace(text)
		if len(note) == 0 {
			return Result{}, fmt.Errorf("текст заметки не должен быть пустым")
		}
		if len([]rune(note)) > 1000 {
			runes := []rune(note)
			note = string(runes[:1000])
		}
		return Result{Observation: fmt.Sprintf("Заметка: %s", note)}, nil

	case "press_key":
		key, err := extractString(input, "key")
		if err != nil {
			return Result{}, err
		}
		if err := t.controller.Press(ctx, "", key); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("Нажата клавиша %s", key)}, nil

	case "hover":
		selector, err := extractString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		if err := t.controller.Hover(ctx, selector); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("навели курсор на %q", selector)}, nil

	case "select_option":
		selector, err := extractString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		value := extractOptionalString(input, "value")
		label := extractOptionalString(input, "label")
		index := extractOptionalInt(input, "index")
		status, err := t.controller.SelectOption(ctx, selector, value, label, index)
		if err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("select_option: %s", status)}, nil

	case "set_checkbox":
		selector, err := extractString(input, "selector")
		if err != nil {
			return Result{}, err
		}
		checked := extractBool(input, "checked")
		status, err := t.controller.SetCheckbox(ctx, selector, checked)
		if err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("set_checkbox: %s", status)}, nil

	case "scroll_page":
		direction := extractOptionalString(input, "direction")
		distance := extractOptionalInt(input, "distance")
		selector := extractOptionalString(input, "selector")
		if err := t.controller.Scroll(ctx, direction, distance, selector); err != nil {
			return Result{}, err
		}
		target := "страницу"
		if strings.TrimSpace(selector) != "" {
			target = fmt.Sprintf("элемент %q", selector)
		}
		return Result{Observation: fmt.Sprintf("Прокрутил %s (%s, %d)", target, direction, distance)}, nil

	case "request_user_input":
		if t.prompt == nil {
			return Result{}, fmt.Errorf("инструмент request_user_input недоступен: не задан обработчик пользовательского ввода")
		}
		message, err := extractString(input, "prompt")
		if err != nil {
			return Result{}, err
		}
		answer, err := t.prompt(ctx, message)
		if err != nil {
			return Result{}, err
		}
		return Result{Observation: answer}, nil

	case "save_storage_state":
		path, err := extractString(input, "path")
		if err != nil {
			return Result{}, err
		}
		if err := t.controller.SaveState(ctx, path); err != nil {
			return Result{}, err
		}
		return Result{Observation: fmt.Sprintf("Состояние браузера сохранено в %s", path)}, nil

	default:
		return Result{}, fmt.Errorf("неизвестный инструмент %q", name)
	}
}

func extractString(input map[string]any, key string) (string, error) {
	value, ok := input[key]
	if !ok {
		return "", fmt.Errorf("требуется поле %q", key)
	}
	switch v := value.(type) {
	case string:
		if strings.TrimSpace(v) == "" {
			return "", fmt.Errorf("поле %q не должно быть пустым", key)
		}
		return v, nil
	case json.Number:
		return v.String(), nil
	default:
		return "", fmt.Errorf("поле %q должно быть строкой", key)
	}
}

func extractOptionalString(input map[string]any, key string) string {
	value, ok := input[key]
	if !ok {
		return ""
	}
	switch v := value.(type) {
	case string:
		return v
	case json.Number:
		return v.String()
	default:
		return ""
	}
}

func extractBool(input map[string]any, key string) bool {
	value, ok := input[key]
	if !ok {
		return false
	}
	switch v := value.(type) {
	case bool:
		return v
	case string:
		return strings.EqualFold(v, "true")
	default:
		return false
	}
}

func extractOptionalInt(input map[string]any, key string) int {
	value, ok := input[key]
	if !ok {
		return 0
	}
	switch v := value.(type) {
	case float64:
		return int(v)
	case int:
		return v
	case int64:
		return int(v)
	case json.Number:
		i, err := v.Int64()
		if err != nil {
			return 0
		}
		return int(i)
	default:
		return 0
	}
}

func extractStringSlice(input map[string]any, key string) ([]string, error) {
	value, ok := input[key]
	if !ok {
		return nil, fmt.Errorf("требуется поле %q", key)
	}
	switch v := value.(type) {
	case []any:
		strings := make([]string, 0, len(v))
		for _, item := range v {
			if s, ok := item.(string); ok {
				strings = append(strings, s)
			} else {
				return nil, fmt.Errorf("поле %q должно быть массивом строк", key)
			}
		}
		return strings, nil
	default:
		return nil, fmt.Errorf("поле %q должно быть массивом строк", key)
	}
}
