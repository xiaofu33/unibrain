document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const docList = document.getElementById('doc-list');
    const convList = document.getElementById('conv-list');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const newChatBtn = document.getElementById('new-chat-btn');
    
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
                const li = document.createElement('li');
                li.innerHTML = `📄 <span>${file.name}</span>`;
                li.style.cssText = "padding: 12px 16px; background: rgba(0, 0, 0, 0.02); border-radius: 8px; margin-bottom: 8px; font-size: 0.9rem; display: flex; align-items: center; gap: 12px;";
                docList.appendChild(li);
                uploadStatus.innerHTML = '✅ <span>解析成功并持久化完毕</span>';
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
                                bubble.innerHTML = marked.parse(fullText);
                                
                                // 处理 a 标签的 target 为 _blank （新窗口打开）
                                bubble.querySelectorAll('a').forEach(a => {
                                    a.setAttribute('target', '_blank');
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

        const avatar = `<div class="avatar">${sender === 'user' ? '我' : 'AI'}</div>`;
        const content = `<div class="bubble">${marked.parse(text)}</div>`;

        msgDiv.innerHTML = sender === 'user' ? content + avatar : avatar + content;
        chatMessages.appendChild(msgDiv);
        
        // 强制滚动到底部，确保新追加的消息可见（尤其是新版 UI 带有渐动效果）
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
                        const li = document.createElement('li');
                        li.innerHTML = `📄 <span>${docName}</span>`;
                        docList.appendChild(li);
                    });
                }
            }
        } catch (error) { console.error(error); }
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

    // 初始化生命周期
    startNewChat();
    loadStoredDocuments();
    refreshConversations();
});
