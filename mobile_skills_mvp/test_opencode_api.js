/**
 * OpenCode API测试脚本
 * 用于测试OpenCode Server的各个API端点
 */

const BASE_URL = 'http://localhost:4096';

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// 颜色：先确保opencode serve正在运行

async function main() {
  console.log('===================================');
  console.log('OpenCode API 测试脚本');
  console.log('===================================');
  console.log('');

  try {
    // 测试1：健康检查
    console.log('[测试1] 健康检查...');
    const healthResponse = await fetch(`${BASE_URL}/global/health`);
    const healthData = await healthResponse.json();
    
    console.log(`健康: ${healthData.healthy}`);
    console.log(`版本: ${healthData.version}`);
    console.log('');

    await sleep(1000);

    // 测试2：创建session
    console.log('[测试2] 创建session...');
    const createSessionResponse = await fetch(`${BASE_URL}/session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        title: 'API测试会话',
      }),
    });
    
    const sessionData = await createSessionResponse.json();
    console.log(`Session ID: ${sessionData.id}`);
    console.log(`标题: ${sessionData.title}`);
    console.log('');

    await sleep(1000);

    // 测试3：发送消息
    console.log('[测试3] 发送消息...');
    const sendMessageResponse = await fetch(`${BASE_URL}/session/${sessionData.id}/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        parts: [
          {
            type: 'text',
            text: '测试消息：请回复"今天学什么？"',
          },
        ],
      }),
    });
    
    const messageData = await sendMessageResponse.json();
    console.log(`消息ID: ${messageData.info.id}`);
    console.log(`角色: ${messageData.info.role}`);
    console.log('');

    await sleep(2000);

    // 测试4：获取消息列表
    console.log('[测试4] 获取消息列表...');
    const getMessagesResponse = await fetch(`${BASE_URL}/session/${sessionData.id}/message`);
    const messagesData = await getMessagesResponse.json();
    
    console.log(`消息数量: ${messagesData.info.length}`);
    console.log('');

    await sleep(1000);

    // 测试5：删除session
    console.log('[测试5] 删除session...');
    const deleteSessionResponse = await fetch(`${BASE_URL}/session/${sessionData.id}`, {
      method: 'DELETE',
    headers: {
        'Content-Type': 'application/json',
      },
    });
    
    const deleteData = await deleteSessionResponse.json();
    console.log(`删除成功: ${deleteData}`);
    console.log('');

    await sleep(1000);

    // 测试6：获取项目信息
    console.log('[测试6] 获取项目信息...');
    const projectResponse = await fetch(`${BASE_URL}/project`);
    const projectData = await projectResponse.json();
    
    console.log(`项目ID: ${projectData.projectID}`);
    console.log(`项目名称: ${projectData.title}`);
    console.log(`工作目录: ${projectData.directory}`);
    console.log('');

    console.log('===================================');
    console.log('所有测试完成！');
    console.log('===================================');
    console.log('');
    console.log('');
    console.log('使用方法：node test.js');
    console.log('');

  } catch (error) {
    console.error('测试失败:', error);
  }
}

// 辅助函数
function fetch(url, options = {}) {
  return fetch(url, options)
    .then(response => response.json())
    .catch(error => {
      throw error;
    });
}

// 运行测试
main();
