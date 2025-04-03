# Абстракции в Computer Use Demo

## 1. Агенты
- **Agent** - базовый класс для всех агентов `agents/agent.py`
  - Поддерживает историю разговора
  - Управляет вызовами API Claude
  - Обрабатывает инструменты и их результаты
- **ManagerAgent** - координатор мульти-агентной системы `agents/manager.py`
  - Декомпозирует задачи
  - Делегирует работу специалистам
  - Может только делать скриншоты
- **SpecialistAgent** - агент для выполнения конкретных задач `agents/specialist.py`
  - Настроен на конкретный домен
  - Может напрямую взаимодействовать с компьютером
- **SpecialistType** - типы специалистов `agents/specialist_types.py`
  - Web Authentication Specialist
  - Lovable Bot Development Specialist
  - General Computer Specialist

## 2. Инструменты
- **BaseAnthropicTool** - базовый абстрактный класс для всех инструментов `tools/base.py`
- **ToolCollection** - коллекция инструментов `tools/collection.py`
  - Инстанцирует и управляет инструментами
  - Предоставляет общий интерфейс для выполнения
- **ToolGroup** - группа инструментов определенной версии `tools/groups.py`
  - Определяет набор классов инструментов для версии API

> **Разница между ToolGroup и ToolCollection:**
> - **ToolGroup** - статическая спецификация набора инструментов; описывает, какие классы инструментов доступны для конкретной версии API
> - **ToolCollection** - рабочий контейнер с созданными экземплярами инструментов, готовыми к использованию агентами

> **Как инструменты подключаются к агенту:**
> 1. При создании агента указывается версия инструментов (например, `tool_version="computer_use_20250124"`)
> 2. Когда агент запускается, он получает нужную `ToolGroup` по этой версии:
>    ```python
>    tool_group = TOOL_GROUPS_BY_VERSION.get(self.tool_version)
>    ```
> 3. Создаёт `ToolCollection` с инструментами из этой группы:
>    ```python
>    tool_collection = ToolCollection(*tool_group.tools, manager_agent=manager)
>    ```
> 4. Когда модель Claude хочет использовать инструмент, агент передаёт вызов в `tool_collection`:
>    ```python
>    result = await tool_collection.run(name=tool_name, tool_input=tool_input)
>    ```

- **Computer Tool** - инструменты взаимодействия с компьютером `tools/computer.py`
  - Управление мышью и клавиатурой
  - Скриншоты и скроллинг
- **Bash Tool** - выполнение bash-команд `tools/bash.py`
- **Edit Tool** - редактирование текстовых файлов `tools/edit.py`
- **Agent Tool** - делегирование задач между агентами `tools/agent_tool.py`
- **Tool Result** - результаты выполнения инструментов `tools/base.py`
  - ToolResult, CLIResult, ToolFailure

## 3. История и отслеживание
- **History** - история разговора для одного агента `agents/history.py`
- **HistoryTree** - древовидная структура для отслеживания всех взаимодействий `history_tree.py`
  - Поддерживает вложенные сессии специалистов
  - Отслеживает вызовы инструментов и их результаты
- **HistoryNode** - узлы в дереве истории `history_tree.py`

## 4. Обработка прерываний и пользовательского ввода

- **Обработка прерываний** - механизм обработки прерываний вызова инструмента
  - **Agent._ensure_history_consistency()** - метод для обработки незавершенных вызовов инструментов
  - **Agent.handle_user_message()** - метод для обработки новых сообщений пользователя
  - **ManagerAgent.handle_user_message()** - перенаправляет сообщения активному агенту
  - **ManagerAgent.get_active_agent()** / **set_active_agent()** - методы для управления активным агентом

- **Жизненный цикл обработки сообщения пользователя**:
  1. Пользователь отправляет сообщение через интерфейс
  2. `ManagerAgent.handle_user_message()` вызывается с этим сообщением
  3. Менеджер определяет активного агента (себя или специалиста)
  4. Активный агент обрабатывает прерывания вызовов инструментов с помощью `_ensure_history_consistency()`
  5. Сообщение добавляется в историю агента и в общее дерево истории
  6. Активный агент запускается с обновленной историей

## 5. Интерфейсы
- **Streamlit Interface** - веб-интерфейс для взаимодействия с системой `streamlit.py`
- **Interface Protocols** - протоколы для избежания циклических зависимостей `interfaces.py`
  - AgentProtocol, HistoryProtocol
- **API Provider** - перечисление провайдеров API Claude `interfaces.py`
  - Anthropic, Bedrock, Vertex 