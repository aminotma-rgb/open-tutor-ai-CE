import { TUTOR_API_BASE_URL } from '$lib/constants';

export type AdaptiveChatMessage = {
	role: 'user' | 'assistant' | 'system';
	content: string;
};

export type AdaptiveChatBody = {
	messages: AdaptiveChatMessage[];
	model?: string;
	stream?: boolean;
	support?: string;
	current_level?: string;
	session_id?: string;
	learning_objectives?: string[];
};

/**
 * POST /api/v1/adaptive/chat
 * Returns [Response, AbortController] — caller reads the SSE stream.
 * Streaming format: `data: {"choices":[{"delta":{"content":"..."}}]}`
 */
export const generateAdaptiveChatCompletion = async (
	token: string,
	body: AdaptiveChatBody
): Promise<[Response | null, AbortController]> => {
	const controller = new AbortController();
	let error = null;

	const res = await fetch(`${TUTOR_API_BASE_URL}/adaptive/chat`, {
		signal: controller.signal,
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		},
		body: JSON.stringify({ stream: true, ...body })
	}).catch((err) => {
		error = err;
		return null;
	});

	if (error) {
		throw error;
	}

	return [res, controller];
};

/**
 * GET /api/v1/adaptive/session/{session_id}
 * Returns the last saved checkpoint for an adaptive session.
 */
export const getAdaptiveSession = async (token: string, sessionId: string) => {
	const res = await fetch(`${TUTOR_API_BASE_URL}/adaptive/session/${sessionId}`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			Authorization: `Bearer ${token}`
		}
	});

	if (!res.ok) {
		throw await res.json();
	}

	return res.json();
};

/**
 * POST /api/v1/adaptive/plan
 * Non-streaming: returns a full AdaptivePlanResponse JSON.
 */
export const runAdaptivePlan = async (
	token: string,
	body: {
		support: string;
		current_level?: string;
		language?: string;
		user_message?: string;
		learning_objectives?: string[];
		feedback_comments?: string[];
		session_id?: string;
	}
) => {
	const res = await fetch(`${TUTOR_API_BASE_URL}/adaptive/plan`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		},
		body: JSON.stringify(body)
	});

	if (!res.ok) {
		throw await res.json();
	}

	return res.json();
};
