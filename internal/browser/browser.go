package browser

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/playwright-community/playwright-go"
)

const (
	defaultNavigationTimeout = 60 * time.Second
	headlessEnv              = "AGENT_HEADLESS"
	defaultScrollDistance    = 600
	defaultWaitTimeout       = 60 * time.Second
)

// Controller описывает управление конкретной страницей/вкладкой браузера.
type Controller interface {
	Close(ctx context.Context) error
	Navigate(ctx context.Context, url string) error
	Click(ctx context.Context, selector string) error
	ClickText(ctx context.Context, text string, exact bool) error
	ClickRole(ctx context.Context, role string, name string, exact bool) error
	Fill(ctx context.Context, selector, text string) error
	Read(ctx context.Context, selector string) (string, error)
	Press(ctx context.Context, selector, key string) error
	WaitFor(ctx context.Context, selector string, timeout time.Duration) error
	Scroll(ctx context.Context, direction string, distance int, selector string) error
	CollectText(ctx context.Context, selector string, attribute string, limit int) ([]string, error)
	CollectLinks(ctx context.Context, selector string, limit int) ([]LinkInfo, error)
	SaveState(ctx context.Context, path string) error
	Hover(ctx context.Context, selector string) error
	SelectOption(ctx context.Context, selector string, value, label string, index int) (string, error)
	SetCheckbox(ctx context.Context, selector string, checked bool) (string, error)
}

type LinkInfo struct {
	Href     string `json:"href"`
	FullHref string `json:"full_href"`
	Text     string `json:"text"`
}

// ModalControl описывает интерактивный элемент внутри модального окна.
// Launcher управляет жизненным циклом браузера Playwright.
type Launcher struct {
	pw       *playwright.Playwright
	browser  playwright.Browser
	headless bool
}

// NewLauncher запускает Playwright и Chromium. В случае отсутствия браузеров
// потребуется установить их командой `npx playwright install`.
func NewLauncher(ctx context.Context) (*Launcher, error) {
	_ = ctx // контекст пригодится позже для тонкой настройки.

	if err := ensureDependencies(); err != nil {
		return nil, err
	}

	pw, err := playwright.Run()
	if err != nil {
		return nil, fmt.Errorf("не удалось запустить Playwright: %w", err)
	}

	headless := parseBoolEnv(headlessEnv, false)

	browser, err := pw.Chromium.Launch(playwright.BrowserTypeLaunchOptions{
		Headless: playwright.Bool(headless),
		Args: []string{
			"--disable-dev-shm-usage",
			"--no-sandbox",
		},
	})
	if err != nil {
		_ = pw.Stop()
		return nil, fmt.Errorf("не удалось запустить Chromium: %w", err)
	}

	return &Launcher{
		pw:       pw,
		browser:  browser,
		headless: headless,
	}, nil
}

// NewController создаёт независимый контекст и страницу без указания сохранённого состояния.
func (l *Launcher) NewController(ctx context.Context) (Controller, error) {
	return l.NewControllerWithState(ctx, "")
}

// NewControllerWithState создаёт контекст и страницу, optionally загружая storage state.
func (l *Launcher) NewControllerWithState(ctx context.Context, storageStatePath string) (Controller, error) {
	_ = ctx

	opts := playwright.BrowserNewContextOptions{
		IgnoreHttpsErrors: playwright.Bool(true),
	}

	if strings.TrimSpace(storageStatePath) != "" {
		opts.StorageStatePath = playwright.String(storageStatePath)
	}

	context, err := l.browser.NewContext(opts)
	if err != nil {
		return nil, fmt.Errorf("не удалось создать контекст браузера: %w", err)
	}

	page, err := context.NewPage()
	if err != nil {
		_ = context.Close()
		return nil, fmt.Errorf("не удалось создать страницу браузера: %w", err)
	}
	page.SetDefaultTimeout(float64(defaultNavigationTimeout.Milliseconds()))

	return &controller{
		context: context,
		page:    page,
	}, nil
}

// Close мягко завершает процесс браузера.
func (l *Launcher) Close() error {
	if l.browser != nil {
		if err := l.browser.Close(); err != nil {
			return err
		}
	}
	if l.pw != nil {
		return l.pw.Stop()
	}
	return nil
}

type controller struct {
	context playwright.BrowserContext
	page    playwright.Page
}

func (c *controller) Close(ctx context.Context) error {
	_ = ctx
	if c.page != nil {
		if err := c.page.Close(); err != nil {
			return err
		}
	}
	if c.context != nil {
		return c.context.Close()
	}
	return nil
}

func (c *controller) Navigate(ctx context.Context, url string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.page == nil {
		return fmt.Errorf("страница не инициализирована")
	}

	// Настраиваем ожидание полной загрузки, чтобы дальнейшие действия опирались на готовый DOM.
	options := playwright.PageGotoOptions{
		WaitUntil: playwright.WaitUntilStateLoad,
		Timeout:   playwright.Float(float64(defaultNavigationTimeout.Milliseconds())),
	}
	_, err := c.page.Goto(url, options)
	return wrapPlaywrightError(err)
}

func (c *controller) Click(ctx context.Context, selector string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.page == nil {
		return fmt.Errorf("страница не инициализирована")
	}
	locator := c.page.Locator(selector)
	if locator == nil {
		return fmt.Errorf("селектор %q не найден", selector)
	}
	// Берём первый видимый элемент, чтобы избежать конфликта strict mode при множественных совпадениях.
	first := locator.First()
	if first == nil {
		return fmt.Errorf("селектор %q не дал кликабельных элементов", selector)
	}
	// Перед кликом убеждаемся, что элемент действительно видим.
	if err := first.WaitFor(playwright.LocatorWaitForOptions{State: playwright.WaitForSelectorStateVisible}); err != nil {
		return wrapPlaywrightError(err)
	}
	return wrapPlaywrightError(first.Click())
}

func (c *controller) ClickText(ctx context.Context, text string, exact bool) (err error) {
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.page == nil {
		return fmt.Errorf("страница не инициализирована")
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("не удалось кликнуть по тексту %q: %v", text, r)
		}
	}()
	options := playwright.PageGetByTextOptions{}
	if exact {
		options.Exact = playwright.Bool(true)
	}
	locator := c.page.GetByText(text, options)
	if locator == nil {
		return fmt.Errorf("элемент с текстом %q не найден", text)
	}
	// Используем первое совпадение; остальные варианты LLM уточнит при необходимости.
	first := locator.First()
	if first == nil {
		return fmt.Errorf("элемент с текстом %q не дал кликабельных совпадений", text)
	}
	// Ждём появления элемента, чтобы не кликать по скрытому тексту.
	if err := first.WaitFor(playwright.LocatorWaitForOptions{
		State: playwright.WaitForSelectorStateVisible,
	}); err != nil {
		return wrapPlaywrightError(err)
	}
	return wrapPlaywrightError(first.Click())
}

func (c *controller) ClickRole(ctx context.Context, role string, name string, exact bool) (err error) {
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.page == nil {
		return fmt.Errorf("страница не инициализирована")
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("не удалось кликнуть по роли %q с именем %q: %v", role, name, r)
		}
	}()

	roleName := strings.ToLower(strings.TrimSpace(role))
	if roleName == "" {
		roleName = "button"
	}
	switch roleName {
	case "button", "link", "menuitem", "checkbox", "radio", "option":
		// supported roles
	default:
		return fmt.Errorf("роль %q не поддерживается", role)
	}
	ariaRole := playwright.AriaRole(roleName)

	opts := playwright.PageGetByRoleOptions{}
	if strings.TrimSpace(name) != "" {
		opts.Name = name
	}
	opts.Exact = playwright.Bool(exact)
	locator := c.page.GetByRole(ariaRole, opts)
	if locator == nil {
		return fmt.Errorf("элемент с ролью %q и именем %q не найден", role, name)
	}
	// Сфокусируемся на первом доступном элементе указанной роли.
	first := locator.First()
	if first == nil {
		return fmt.Errorf("элемент с ролью %q и именем %q не найден среди кликабельных совпадений", role, name)
	}
	// Ждём появления элемента, иначе Playwright вернёт ошибку по скрытым узлам.
	if err := first.WaitFor(playwright.LocatorWaitForOptions{
		State: playwright.WaitForSelectorStateVisible,
	}); err != nil {
		return wrapPlaywrightError(err)
	}
	return wrapPlaywrightError(first.Click())
}

func (c *controller) Fill(ctx context.Context, selector, text string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.page == nil {
		return fmt.Errorf("страница не инициализирована")
	}
	locator := c.page.Locator(selector)
	if locator == nil {
		return fmt.Errorf("селектор %q не найден", selector)
	}
	// Просим Playwright дождаться появления элемента.
	if err := locator.WaitFor(playwright.LocatorWaitForOptions{
		State: playwright.WaitForSelectorStateVisible,
	}); err != nil {
		return wrapPlaywrightError(err)
	}
	// Лёгкий клик помогает сфокусировать поле, особенно если это contenteditable.
	_ = locator.Click()

	if err := locator.Fill(text); err != nil {
		// Запасной путь: выделяем текст и печатаем вручную.
		_ = locator.Press("Control+A")
		_ = locator.Press("Meta+A")
		_ = locator.Press("Backspace")
		if typeErr := locator.Type(text); typeErr != nil {
			return wrapPlaywrightError(err)
		}
	}
	return nil
}

func (c *controller) Read(ctx context.Context, selector string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if c.page == nil {
		return "", fmt.Errorf("страница не инициализирована")
	}
	if strings.TrimSpace(selector) == "" {
		value, err := c.page.Evaluate(`() => document.body.innerText`)
		if err != nil {
			return "", wrapPlaywrightError(err)
		}
		if text, ok := value.(string); ok {
			return text, nil
		}
		return fmt.Sprintf("%v", value), nil
	}

	locator := c.page.Locator(selector)
	if locator == nil {
		return "", fmt.Errorf("селектор %q не найден", selector)
	}

	text, err := locator.InnerText()
	if err != nil {
		return "", wrapPlaywrightError(err)
	}
	return text, nil
}

func (c *controller) Press(ctx context.Context, selector, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.page == nil {
		return fmt.Errorf("страница не инициализирована")
	}
	if strings.TrimSpace(selector) == "" {
		return wrapPlaywrightError(c.page.Keyboard().Press(key))
	}

	locator := c.page.Locator(selector)
	if locator == nil {
		return fmt.Errorf("селектор %q не найден", selector)
	}
	return wrapPlaywrightError(locator.Press(key))
}

func (c *controller) WaitFor(ctx context.Context, selector string, timeout time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.page == nil {
		return fmt.Errorf("страница не инициализирована")
	}
	if timeout <= 0 {
		timeout = defaultWaitTimeout
	}
	locator := c.page.Locator(selector)
	if locator == nil {
		return fmt.Errorf("селектор %q не найден", selector)
	}
	options := playwright.LocatorWaitForOptions{}
	if timeout > 0 {
		options.Timeout = playwright.Float(timeout.Seconds() * 1000)
	}
	return wrapPlaywrightError(locator.WaitFor(options))
}

func (c *controller) Scroll(ctx context.Context, direction string, distance int, selector string) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	if strings.TrimSpace(selector) != "" {
		locator := c.page.Locator(selector)
		if locator == nil {
			return fmt.Errorf("селектор %q не найден", selector)
		}
		return wrapPlaywrightError(locator.ScrollIntoViewIfNeeded())
	}

	if distance == 0 {
		distance = defaultScrollDistance
	}
	move := distance

	switch strings.ToLower(direction) {
	case "up", "north":
		move = -distance
	case "top":
		_, err := c.page.Evaluate("window.scrollTo(0, 0);")
		return wrapPlaywrightError(err)
	case "bottom":
		_, err := c.page.Evaluate("window.scrollTo(0, document.body.scrollHeight);")
		return wrapPlaywrightError(err)
	case "page_up":
		move = -distance * 2
	case "page_down":
		move = distance * 2
	}

	script := fmt.Sprintf("window.scrollBy(0, %d);", move)
	_, err := c.page.Evaluate(script)
	return wrapPlaywrightError(err)
}

func (c *controller) CollectText(ctx context.Context, selector string, attribute string, limit int) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if c.page == nil {
		return nil, fmt.Errorf("страница не инициализирована")
	}

	locator := c.page.Locator(selector)
	if locator == nil {
		return nil, fmt.Errorf("селектор %q не найден", selector)
	}

	countRaw, err := locator.Count()
	if err != nil {
		return nil, wrapPlaywrightError(err)
	}
	count := int(countRaw)

	if limit > 0 && count > limit {
		count = limit
	}

	results := make([]string, 0, count)
	for i := 0; i < count; i++ {
		item := locator.Nth(i)
		if attribute != "" {
			attr, attrErr := item.GetAttribute(attribute)
			if attrErr != nil {
				return nil, wrapPlaywrightError(attrErr)
			}
			if attr != "" {
				results = append(results, attr)
			}
			continue
		}

		text, textErr := item.InnerText()
		if textErr != nil {
			return nil, wrapPlaywrightError(textErr)
		}
		results = append(results, text)
	}

	return results, nil
}

func (c *controller) CollectLinks(ctx context.Context, selector string, limit int) ([]LinkInfo, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if c.page == nil {
		return nil, fmt.Errorf("страница не инициализирована")
	}

	query := strings.TrimSpace(selector)
	if query != "" {
		return c.collectLinksBySelector(ctx, query, limit)
	}

	fallbacks := []string{
		"a[href]",
		"[role='link']",
		"[tabindex][onclick]",
		"[data-action*='link']",
	}

	for _, candidate := range fallbacks {
		links, err := c.collectLinksBySelector(ctx, candidate, limit)
		if err != nil {
			continue
		}
		if len(links) > 0 {
			return links, nil
		}
	}

	return nil, fmt.Errorf("не удалось найти ссылки по универсальным селекторам")
}

func (c *controller) collectLinksBySelector(ctx context.Context, selector string, limit int) ([]LinkInfo, error) {
	loc := c.page.Locator(selector)
	if loc == nil {
		return nil, fmt.Errorf("селектор %q не найден", selector)
	}

	countRaw, err := loc.Count()
	if err != nil {
		return nil, wrapPlaywrightError(err)
	}
	count := int(countRaw)
	if count == 0 {
		return []LinkInfo{}, nil
	}
	if limit > 0 && count > limit {
		count = limit
	}

	links := make([]LinkInfo, 0, count)

	pageURL := c.page.URL()
	base, baseErr := url.Parse(pageURL)
	if baseErr != nil {
		base = &url.URL{}
	}

	for i := 0; i < count; i++ {
		item := loc.Nth(i)
		href, err := item.GetAttribute("href")
		if err != nil {
			return nil, wrapPlaywrightError(err)
		}
		linkHref := strings.TrimSpace(href)

		if linkHref == "" {
			// Запускаем скрипт, который переберёт атрибуты ссылки и найдёт подходящий адрес.
			script := `(el) => {
				const attrs = el.attributes || [];
				for (const attr of attrs) {
					if (/href|data-href|data-link|data-url/.test(attr.name)) {
						return attr.value;
					}
				}
				return '';
			}`
			res, evalErr := item.Evaluate(script, nil)
			if evalErr == nil {
				if str, ok := res.(string); ok {
					linkHref = strings.TrimSpace(str)
				}
			}
		}

		text, err := item.InnerText()
		if err != nil {
			return nil, wrapPlaywrightError(err)
		}
		linkText := strings.TrimSpace(text)

		if linkHref == "" && linkText == "" {
			continue
		}

		fullHref := linkHref
		if linkHref != "" {
			parsed, parseErr := url.Parse(linkHref)
			if parseErr == nil {
				full := base.ResolveReference(parsed)
				fullHref = full.String()
			}
		}

		links = append(links, LinkInfo{Href: linkHref, FullHref: fullHref, Text: linkText})
	}

	return links, nil
}

func (c *controller) SaveState(ctx context.Context, path string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	state, err := c.context.StorageState()
	if err != nil {
		return wrapPlaywrightError(err)
	}
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("не удалось сериализовать storage state: %w", err)
	}
	return os.WriteFile(path, data, 0o600)
}

func (c *controller) Hover(ctx context.Context, selector string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if c.page == nil {
		return fmt.Errorf("страница не инициализирована")
	}
	locator := c.page.Locator(selector)
	if locator == nil {
		return fmt.Errorf("селектор %q не найден", selector)
	}
	return wrapPlaywrightError(locator.Hover())
}

func (c *controller) SelectOption(ctx context.Context, selector string, value, label string, index int) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if c.page == nil {
		return "", fmt.Errorf("страница не инициализирована")
	}
	// Скрипт работает внутри страницы и повторяет логику выбора опции пользователем.
	script := `(selector, payload) => {
		const el = document.querySelector(selector);
		if (!el) return 'not_found';
		if (!(el instanceof HTMLSelectElement)) return 'not_select';
		const setValue = (opt) => {
			if (!opt) return false;
			el.value = opt.value;
			return true;
		};
		let changed = false;
		if (payload.value) {
			const opt = Array.from(el.options || []).find(o => o.value === payload.value);
			changed = setValue(opt);
		} else if (payload.label) {
			const opt = Array.from(el.options || []).find(o => o.label === payload.label || o.text === payload.label);
			changed = setValue(opt);
		} else if (typeof payload.index === 'number') {
			const opt = el.options && el.options[payload.index];
			changed = setValue(opt);
		}
		if (!changed) return 'option_not_found';
		el.dispatchEvent(new Event('input', { bubbles: true }));
		el.dispatchEvent(new Event('change', { bubbles: true }));
		return 'ok';
	}`
	args := map[string]any{
		"value": value,
		"label": label,
		"index": index,
	}
	res, err := c.page.Evaluate(script, selector, args)
	if err != nil {
		return "", wrapPlaywrightError(err)
	}
	if str, ok := res.(string); ok {
		if str != "ok" {
			return str, fmt.Errorf("select_option: %s", str)
		}
		return str, nil
	}
	return "", nil
}

func (c *controller) SetCheckbox(ctx context.Context, selector string, checked bool) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if c.page == nil {
		return "", fmt.Errorf("страница не инициализирована")
	}
	locator := c.page.Locator(selector)
	if locator == nil {
		return "", fmt.Errorf("селектор %q не найден", selector)
	}

	if err := locator.WaitFor(playwright.LocatorWaitForOptions{State: playwright.WaitForSelectorStateVisible}); err != nil {
		return "", wrapPlaywrightError(err)
	}

	if checked {
		if err := locator.Check(); err != nil {
			return "", wrapPlaywrightError(err)
		}
		return "checkbox установлена", nil
	}

	if err := locator.Uncheck(); err != nil {
		return "", wrapPlaywrightError(err)
	}
	return "checkbox снята", nil
}

func wrapPlaywrightError(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("ошибка Playwright: %w", err)
}

func parseBoolEnv(name string, defaultValue bool) bool {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return defaultValue
	}
	switch strings.ToLower(value) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return defaultValue
	}
}

func ensureDependencies() error {
	// В большинстве окружений браузеры уже установлены. Здесь можно добавить
	// проверку и установку при необходимости. Пока что просто возвращаем nil.
	return nil
}
