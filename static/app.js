document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const docList = document.getElementById('doc-list');
    const convList = document.getElementById('conv-list');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const newChatBtn = document.getElementById('new-chat-btn');
    const previewModal = document.getElementById('preview-modal');
    const previewFrame = document.getElementById('preview-frame');
    const previewFilename = document.getElementById('preview-filename');
    
    let currentSessionId = null;

    // 清空聊天窗口并重置会话状态
    function startNewChat() {
        currentSessionId = null;
        chatMessages.innerHTML = '';
        appendMessage('ai', '您好！我是企业制度管理助手。我已经读取了目前的制度库。请问您有什么关于制度的问题想要查询？');
        document.querySelectorAll('#conv-list li').forEach(el => el.classList.remove('active'));
    }

    // 初始化事件绑定
    newChatBtn.addEventListener('click', startNewChat);

    // 文件上传逻辑
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        uploadStatus.classList.remove('hidden');
        uploadStatus.innerHTML = '<div class="spinner"></div><span>处理并向量化...</span>';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/v1/documents/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                appendDocToList(file.name);
                uploadStatus.innerHTML = '✅ <span>解析成功</span>';
            } else {
                uploadStatus.innerHTML = `❌ <span>失败: ${data.detail}</span>`;
            }
        } catch (error) {
            uploadStatus.innerHTML = `❌ <span>网络异常</span>`;
        }

        setTimeout(() => { uploadStatus.classList.add('hidden'); }, 3000);
        fileInput.value = ''; 
    });

    // 文字问答交互
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;

        appendMessage('user', text);
        chatInput.value = '';
        
        const typingId = appendMessage('ai', '思考中...', true);

        try {
            const res = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: text,
                    session_id: currentSessionId
                })
            });
            
            // 将原先正在 loading 的气泡清空，准备逐渐放入流式字符
            const responseDiv = document.getElementById(typingId);
            responseDiv.innerHTML = `<div class="avatar">AI</div><div class="bubble"></div>`;
            const bubble = responseDiv.querySelector('.bubble');

            if (!res.ok) {
                const data = await res.json();
                bubble.innerHTML = escapeHtml(`请求失败: ${data.detail || '未知错误'}`);
                return;
            }

            // 处理打字机流式逐字读取 (SSE)
            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let done = false;
            let fullText = "";
            let buffer = "";

            while (!done) {
                const { value, done: readerDone } = await reader.read();
                done = readerDone;
                if (value) {
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n\n'); 
                    buffer = lines.pop(); // 拿到可能尚未读取完全的数据截断面，留在下一次使用
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.substring(6);
                            try {
                                const parsed = JSON.parse(dataStr);
                                if (parsed.session_id && !currentSessionId) {
                                    currentSessionId = parsed.session_id;
                                    refreshConversations();
                                }
                                fullText += parsed.delta;
                                // 使用 marked 解析并将 [来源: XXX] 替换为带样式的引用链接
                                let htmlContent = marked.parse(fullText);
                                htmlContent = htmlContent.replace(/\[来源[:：]\s*(.+?)\]/g, (match, filename) => {
                                    const safeName = encodeURIComponent(filename.trim());
                                    return `<a class="citation-link" href="/static/uploads/${safeName}" data-preview="true">${match}</a>`;
                                });
                                bubble.innerHTML = htmlContent;
                                
                                // 处理 a 标签的点击监听
                                bubble.querySelectorAll('a').forEach(a => {
                                    if (a.dataset.preview === "true" || a.href.includes('/static/uploads/')) {
                                        a.addEventListener('click', (e) => {
                                            e.preventDefault();
                                            const url = a.getAttribute('href');
                                            const filename = a.innerText.replace(/\[|\]|来源[:：]\s*/g, '') || '文件预览';
                                            openPreview(url, filename);
                                        });
                                    } else {
                                        a.setAttribute('target', '_blank');
                                    }
                                });
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            } catch(e) { }
                        }
                    }
                }
            }
            
        } catch (error) {
            document.getElementById(typingId).remove();
            appendMessage('ai', `接口请求异常: ${error.message} (可能遇到大模型速率限制或网络断开)`);
        }
    });

    // 工具：往 UI 追加消息气泡
    function appendMessage(sender, text, isTyping = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        const id = 'msg-' + Date.now();
        if (isTyping) msgDiv.id = id;

        const avatarInitial = sender === 'user' ? 'U' : 'AI';
        const avatar = `<div class="avatar">${avatarInitial}</div>`;
        const content = `<div class="bubble">${marked.parse(text)}</div>`;

        msgDiv.innerHTML = sender === 'user' ? content + avatar : avatar + content;
        chatMessages.appendChild(msgDiv);
        
        // 强制滚动到底部
        requestAnimationFrame(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });
        
        return id;
    }

    // ========== 历史记录架构 ==========

    // 1. 初始化拉取文档列表（兼容旧逻辑）
    async function loadStoredDocuments() {
        try {
            const res = await fetch('/api/v1/documents');
            if (res.ok) {
                const data = await res.json();
                if (data.documents && data.documents.length > 0) {
                    data.documents.forEach(docName => {
                        appendDocToList(docName);
                    });
                }
            }
        } catch (error) { console.error(error); }
    }

    // 1.1 封装：向列表添加文档项内容封装完毕。
    function appendDocToList(docName) {
        const li = document.createElement('li');
        li.innerHTML = `
            <div class="doc-item">
                <span class="doc-name">📄 ${docName}</span>
                <button class="delete-doc-btn" title="删除该文档">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                </button>
            </div>
        `;
        
        const deleteBtn = li.querySelector('.delete-doc-btn');
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (confirm(`确定要从知识库中删除 "${docName}" 吗？此操作不可恢复。`)) {
                deleteDocument(docName, li);
            }
        });

        docList.appendChild(li);
    }

    async function deleteDocument(docName, liElement) {
        try {
            const res = await fetch(`/api/v1/documents/${encodeURIComponent(docName)}`, {
                method: 'DELETE'
            });
            if (res.ok) {
                liElement.style.opacity = '0';
                liElement.style.transform = 'translateX(-20px)';
                setTimeout(() => liElement.remove(), 300);
            } else {
                const data = await res.json();
                alert(`删除失败: ${data.detail}`);
            }
        } catch (error) {
            alert('删除异常，请检查网络内容封装完毕。');
        }
    }

    // 2. 拉取历史会话列表
    async function refreshConversations() {
        try {
            const res = await fetch('/api/v1/conversations');
            if (res.ok) {
                const data = await res.json();
                convList.innerHTML = '';
                if (data.conversations) {
                    data.conversations.forEach(conv => {
                        const li = document.createElement('li');
                        li.innerHTML = `💬 <span>${conv.title || "新对话"}</span>`;
                        li.onclick = () => loadMessagesForSession(conv.id, li);
                        if(currentSessionId === conv.id) li.classList.add('active');
                        convList.appendChild(li);
                    });
                }
            }
        } catch (error) { console.error(error); }
    }

    // 3. 用户点击左侧历史，右侧恢复聊天流
    async function loadMessagesForSession(sessionId, liElement) {
        currentSessionId = sessionId;
        document.querySelectorAll('#conv-list li').forEach(el => el.classList.remove('active'));
        if(liElement) liElement.classList.add('active');
        
        chatMessages.innerHTML = '';
        try {
            const res = await fetch(`/api/v1/conversations/${sessionId}/messages`);
            const data = await res.json();
            if (res.ok && data.messages.length > 0) {
                data.messages.forEach(msg => {
                    appendMessage(msg.sender, msg.content);
                });
            } else {
                startNewChat();
            }
        } catch (error) {
            console.error("加载消息失败", error);
        }
    }

    // ========== 主题切换逻辑 ==========
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    
    const sunIcon = `<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>`;
    const moonIcon = `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>`;

    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        themeIcon.innerHTML = theme === 'light' ? moonIcon : sunIcon;
        localStorage.setItem('theme', theme);
    }

    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);

    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        setTheme(currentTheme === 'light' ? 'dark' : 'light');
    });

    // ========== 预览功能 ==========
    function openPreview(url, filename) {
        previewFrame.src = url;
        previewFilename.innerText = filename;
        previewModal.classList.add('active');
        // 禁止背景滚动
        document.body.style.overflow = 'hidden';
    }

    window.closePreview = function() {
        previewModal.classList.remove('active');
        previewFrame.src = "";
        document.body.style.overflow = '';
    };

    // 点击蒙层关闭
    previewModal.addEventListener('click', (e) => {
        if (e.target === previewModal) closePreview();
    });

    // ESC 键关闭
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && previewModal.classList.contains('active')) {
            closePreview();
        }
    });

    // 初始化生命周期
    startNewChat();
    loadStoredDocuments();
    refreshConversations();
});
